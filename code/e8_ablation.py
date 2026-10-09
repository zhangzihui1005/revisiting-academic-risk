import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
from pathlib import Path
import argparse,subprocess,concurrent.futures,time
import numpy as np,pandas as pd,joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import e6_common as c
ROOT=Path(__file__).resolve().parent;T=ROOT.parent/'E4T_OULAD_temporal_generalization_v1.0';sys.path.insert(0,str(T));import run as t
NAMES=['LR','CatBoost'];EXPS={28:'Day28_failure',56:'E4T-B_day56_failure_secondary'}
def configuration():return c.load(ROOT/'jobs.json')
def prepare():
 assert not (ROOT/'jobs.json').exists();groups=c.load(ROOT/'feature_groups.json');cfg=c.load(T/'config.json');features=cfg[EXPS[28]]['features'];flat=[f for v in groups.values() for f in v];assert len(flat)==len(set(flat))==16 and set(flat)==set(features);jobs={}
 for h,ex in EXPS.items():
  for n in NAMES:
   for group,remove in groups.items():
    key=f'Day{h}_{n}_minus_{group}';dest=ROOT/key;dest.mkdir();jobs[key]=dict(horizon=h,experiment=ex,family=n,group=group,features=[f for f in features if f not in remove],params=c.load(T/ex/n/'selection.json')['params'])
 c.dump(ROOT/'jobs.json',jobs);c.dump(ROOT/'training_manifest.json',dict(code_sha256=c.sha(Path(__file__)),protocol_sha256=c.sha(ROOT/'PROTOCOL.txt'),groups_sha256=c.sha(ROOT/'feature_groups.json'),jobs_sha256=c.sha(ROOT/'jobs.json'),final_models=16,source_fits=144,new_Optuna_trials=0))
def worker(key):
 manifest=c.load(ROOT/'training_manifest.json');assert c.sha(Path(__file__))==manifest['code_sha256'];job=configuration()[key];ex=job['experiment'];cfg=c.load(T/'config.json')[ex];data=t.data(cfg);features=job['features'];dest=ROOT/key;parts=[]
 for g in cfg['folds']:
  tr=data.loc[t.members(ex,'inner_'+g+'_train')];va=data.loc[t.members(ex,'inner_'+g+'_validation')];assert not set(tr.id)&set(va.id);m=t.model(job['family'],job['params'],dict(features=features,categorical=[]));m.fit(tr[features],tr.y);p=m.predict_proba(va[features])[:,1].astype(float);o=va[['key','row_id','id','domain','module','y']].copy();o['validation_domain']=g;o['probability']=p;parts.append(o)
 oof=pd.concat(parts,ignore_index=True);oof.to_csv(dest/'source_oof.csv.gz',index=False);threshold,curve=t.threshold_curve(oof);curve.to_csv(dest/'source_threshold_curve.csv.gz',index=False)
 tr=data.loc[t.members(ex,'train')];m=t.model(job['family'],job['params'],dict(features=features,categorical=[]));m.fit(tr[features],tr.y);joblib.dump(m,dest/'model.joblib',compress=3);c.dump(dest/'selection.json',dict(**job,threshold=threshold,model_sha256=c.sha(dest/'model.joblib'),source_oof_sha256=c.sha(dest/'source_oof.csv.gz')));print('FIT',key,flush=True)
def train():
 def one(key):
  if not (ROOT/key/'selection.json').exists():
   with (ROOT/key/'train.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-u',str(Path(__file__)),'worker','--job',key],stdout=f,stderr=subprocess.STDOUT,check=True)
  return key
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(one,k) for k in configuration()]
  for i,f in enumerate(concurrent.futures.as_completed(fs),1):print('DONE',i,16,f.result(),flush=True)
 c.dump(ROOT/'models_locked.json',dict(files={str(p.relative_to(ROOT)).replace('\\','/'):c.sha(p) for k in configuration() for p in (ROOT/k).iterdir() if p.name in ['selection.json','model.joblib','source_oof.csv.gz']},locked_before_target_scoring=True))
def score():
 for p,h in c.load(ROOT/'models_locked.json')['files'].items():assert c.sha(ROOT/p)==h
 rows=[];domain=[];source=[]
 for key,j in configuration().items():
  ex=j['experiment'];cfg=c.load(T/'config.json')[ex];d=t.data(cfg);te=d.loc[t.members(ex,'test')];m=joblib.load(ROOT/key/'model.joblib');p=m.predict_proba(te[j['features']])[:,1].astype(float);z=te[['key','row_id','id','domain','module','y']].copy();z['probability']=p;z.to_csv(ROOT/key/'target_predictions.csv',index=False);meta=dict(experiment=key,horizon=j['horizon'],family=j['family'],group=j['group']);rows.append(dict(**meta,**c.metrics(z.y,z.probability)))
  for g,v in z.groupby('module'):domain.append(dict(**meta,module=g,**c.metrics(v.y,v.probability)))
  o=pd.read_csv(ROOT/key/'source_oof.csv.gz',float_precision='round_trip')
  for g,v in o.groupby('domain'):source.append(dict(**meta,domain=g,**c.metrics(v.y,v.probability)))
 for h,ex in EXPS.items():
  for n in NAMES:
   z=pd.read_csv(T/ex/n/'test_predictions.csv',float_precision='round_trip');rows.append(dict(experiment=ex,horizon=h,family=n,group='Full',**c.metrics(z.y,z.probability)))
 pd.DataFrame(rows).to_csv(ROOT/'target_metrics.csv',index=False);pd.DataFrame(domain).to_csv(ROOT/'target_module_metrics.csv',index=False);pd.DataFrame(source).to_csv(ROOT/'source_domain_metrics.csv',index=False)
def cache(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p);o=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[o])!=0),len(p)-1];return y,p,o,ends
def ap_nap(a,w):
 y,p,o,ends=a;total=w.sum();pos=w@y
 if pos==0 or pos==total:return np.array([np.nan,np.nan])
 tp=np.cumsum((w*y)[o])[ends];nn=np.cumsum(w[o])[ends];prec=np.divide(tp,nn,out=np.zeros(len(tp)),where=nn>0);ap=np.diff(np.r_[0.,tp/pos])@prec;pi=pos/total;return np.array([ap,(ap-pi)/(1-pi)])
def analyze():
 out=[];fixed=[]
 for n in NAMES:
  for group in c.load(ROOT/'feature_groups.json'):
   preds=[]
   for h in [28,56]:
    for variant in ['Full',group]:
     path=T/EXPS[h]/n/'test_predictions.csv' if variant=='Full' else ROOT/f'Day{h}_{n}_minus_{group}'/'target_predictions.csv';z=pd.read_csv(path,float_precision='round_trip');z=z[['key','id','y','probability']].rename(columns={'probability':f'p{h}_{variant}'});preds.append(z)
   z=preds[0]
   for p in preds[1:]:z=z.merge(p,on=['key','id','y'],validate='one_to_one')
   z=z.sort_values('key').reset_index(drop=True);cols=[f'p{h}_{v}' for h in [28,56] for v in ['Full',group]];aa=[cache(z.y,z[col]) for col in cols];point=np.vstack([ap_nap(a,np.ones(len(z))) for a in aa]);_,inv=np.unique(z.id,return_inverse=True);ns=inv.max()+1;rng=np.random.default_rng(c.SEED);reps=[]
   for i in range(1000):
    w=np.bincount(rng.integers(0,ns,ns),minlength=ns)[inv];v=np.vstack([ap_nap(a,w) for a in aa]);reps.append(np.r_[v[1]-v[0],v[3]-v[2],(v[3]-v[1])-(v[2]-v[0])])
    if i<3:
     idx=np.repeat(np.arange(len(z)),w)
     for a,vv in zip(aa,v):rr=c.metrics(a[0][idx],a[1][idx]);assert np.allclose(vv,[rr['AP'],rr['nAP']],atol=1e-10)
   reps=pd.DataFrame(reps,columns=['day28_delta_AP','day28_delta_nAP','day56_delta_AP','day56_delta_nAP','gain_change_AP','gain_change_nAP']);dest=ROOT/'paired'/n/group;dest.mkdir(parents=True);z.to_csv(dest/'common_predictions.csv',index=False);reps.to_csv(dest/'bootstrap.csv',index=False)
   full=point[2]-point[0];abl=point[3]-point[1];row=dict(family=n,removed_group=group,N=len(z),students=ns,Full_gain_AP=full[0],Full_gain_nAP=full[1],Reduced_gain_AP=abl[0],Reduced_gain_nAP=abl[1],gain_change_AP=abl[0]-full[0],gain_change_nAP=abl[1]-full[1])
   for met in ['AP','nAP']:
    lo,hi=reps['gain_change_'+met].quantile([.025,.975]);row.update({f'gain_change_{met}_CI_low':lo,f'gain_change_{met}_CI_high':hi})
   out.append(row)
   for h,i in [(28,0),(56,2)]:
    row=dict(family=n,horizon=h,removed_group=group,N=len(z))
    for k,met in enumerate(['AP','nAP']):lo,hi=reps[f'day{h}_delta_{met}'].quantile([.025,.975]);row.update({f'Full_{met}':point[i,k],f'Reduced_{met}':point[i+1,k],f'delta_{met}':point[i+1,k]-point[i,k],f'delta_{met}_CI_low':lo,f'delta_{met}_CI_high':hi})
    fixed.append(row)
 r=pd.DataFrame(out);r.to_csv(ROOT/'information_gain_ablation.csv',index=False);pd.DataFrame(fixed).to_csv(ROOT/'fixed_horizon_ablation.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
 for ax,n in zip(axes,NAMES):
  q=r[r.family.eq(n)];x=np.arange(len(q));ax.errorbar(x,q.gain_change_nAP,yerr=[q.gain_change_nAP-q.gain_change_nAP_CI_low,q.gain_change_nAP_CI_high-q.gain_change_nAP],fmt='o',capsize=4);ax.axhline(0,color='gray',ls='--');ax.set(xticks=x,xticklabels=[s.split('_')[0] for s in q.removed_group],ylabel='Change in Day28→56 nAP gain',title=n+' / remove one group');ax.grid(axis='y',alpha=.2)
 fig.savefig(ROOT/'information_gain_ablation.png',dpi=180);fig.savefig(ROOT/'information_gain_ablation.pdf');plt.close(fig)
 c.dump(ROOT/'execution_checks.json',dict(status='passed',new_Optuna_trials=0,source_fits=144,final_models=16,common_test_pairs=8,bootstrap_repeats=1000,first_three_replicates_explicitly_verified=True,targets_scored_after_all_models_locked=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','worker']);p.add_argument('--job');a=p.parse_args()
 if a.stage=='worker':worker(a.job)
 else:prepare();train();score();analyze()
