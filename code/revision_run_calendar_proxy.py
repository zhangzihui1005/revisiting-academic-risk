"""E12 proxy-only training: redo all source development on eligible 2013B domains."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='6'
from pathlib import Path
import json,time,datetime,concurrent.futures,zipfile,hashlib
import numpy as np,pandas as pd,optuna,joblib
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];A=R/'archived_E1_E9/outputs';T=A/'E4T_OULAD_temporal_generalization_v1.0';OUT=R/'experiments_new/E12_calendar_audit';FAM=['LR','XGBoost','CatBoost']
sys.path.insert(0,str(A/'experiment1_day28'));import run_experiment as e1
sys.path.insert(0,str(R/'archived_E1_E9/work'));import run_e6c as cal
def read(p):return pd.read_csv(p,float_precision='round_trip')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def kh(k):return hashlib.sha256(('\n'.join(sorted(k))+'\n').encode()).hexdigest()
def threshold(o):
 cand=np.unique(np.r_[0.,o.p.to_numpy(),1.]);scores=[]
 for _,v in o.groupby('validation_domain'):
  order=np.argsort(v.p,kind='stable');p=v.p.to_numpy()[order];y=v.y.to_numpy()[order];pos=np.searchsorted(p,cand,side='left');cum=np.r_[0,np.cumsum(y)];tp=cum[-1]-cum[pos];scores.append(2*tp/np.maximum(y.sum()+len(y)-pos,1))
 return float(cand[np.argmax(np.mean(scores,axis=0))])
def prep():
 with zipfile.ZipFile(R/'archived_E1_E9/raw_data/OULAD DATASET.zip') as z:info=read(z.open('studentInfo.csv'))
 info['domain']=info.code_module+'_'+info.code_presentation
 jobs=[];aud=[];blacklist=set(info.loc[info.code_presentation.str.startswith('2014'),'id_student'])
 for ex in ['Day28_failure','Day28_withdrawal','E4T-B_day56_failure_secondary']:
  d=read(T/ex/'data.csv').set_index('key',drop=False);orig=read(T/ex/'train.csv');keys=orig.loc[orig.domain.str.endswith('2013B'),'key'];tr=d.loc[keys].sort_index();assert not set(tr.id)&blacklist
  dest=OUT/'proxy_inputs'/ex;dest.mkdir(parents=True,exist_ok=True);tr.to_csv(dest/'source.csv',index=False)
  for domain in sorted(tr.domain.unique()):
   held=set(info.loc[info.domain.eq(domain),'id_student']);it=tr[~tr.domain.eq(domain)&~tr.id.isin(held)];iv=tr[tr.domain.eq(domain)];assert not set(it.id)&set(iv.id)
   it.to_csv(dest/f'inner_{domain}_train.csv',index=False);iv.to_csv(dest/f'inner_{domain}_validation.csv',index=False);aud.append(dict(experiment=ex,validation_domain=domain,train_N=len(it),validation_N=len(iv),student_overlap=0,raw_2014_overlap=0))
  for f in FAM:jobs.append(dict(experiment=ex,family=f,source=str(dest/'source.csv'),folds=sorted(tr.domain.unique()),trials=15 if 'day56' in ex else 20,source_keys_hash=kh(tr.key),dest=str(OUT/'proxy_fits'/ex/f)))
 pd.DataFrame(aud).to_csv(OUT/'E12_proxy_inner_source_audit.csv',index=False)
 plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),label='PROXY calendar sensitivity; release timestamps unverified',release_lags=[0,30,60],source_policy='2014B: 2013B only; 2014J: all nine original source domains',all_three_lags_yield_same_source_sets=True,selection='equal-three-source-domain mean AP; tie order LR,XGBoost,CatBoost',calibration='new positive-slope Platt on selected-trial source OOF; equal records; OOF reused after tuning and not fully nested',search='original frozen E1 ranges, fresh studies; no target labels in tuning',jobs=jobs)
 p=OUT/'proxy_prerun_plan.json'
 if p.exists():assert load(p)['jobs']==jobs
 else:dump(p,plan)
 return jobs
def worker(j):
 out=Path(j['dest']);out.mkdir(parents=True,exist_ok=True)
 if (out/'selection.json').exists():return j['experiment']+' '+j['family']+' cached'
 start=time.time();src=Path(j['source']);tr=read(src);assert kh(tr.key)==j['source_keys_hash'];parts=[(f,read(src.parent/f'inner_{f}_train.csv'),read(src.parent/f'inner_{f}_validation.csv')) for f in j['folds']]
 optuna.logging.set_verbosity(optuna.logging.WARNING);study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=e1.SEED),study_name='E12_'+j['experiment']+'_'+j['family'],storage='sqlite:///'+str(out/'study.sqlite3').replace('\\','/'),load_if_exists=True)
 def score(params,save=False):
  aps=[];oo=[]
  for domain,it,iv in parts:
   m=e1.model(j['family'],params)
   with threadpool_limits(limits=6):m.fit(it[e1.FEATURES],it.y);p=m.predict_proba(iv[e1.FEATURES])[:,1].astype(float)
   aps.append(average_precision_score(iv.y,p))
   if save:z=iv[['key','id','domain','y']].copy();z['p']=p;z['validation_domain']=domain;oo.append(z)
  return np.mean(aps),pd.concat(oo,ignore_index=True) if save else None
 study.optimize(lambda trial:score(e1.parameters(trial,j['family']))[0],n_trials=max(0,j['trials']-len(study.trials)))
 value,oof=score(study.best_params,True);assert abs(value-study.best_value)<1e-10;oof.to_csv(out/'source_oof.csv',index=False)
 th=threshold(oof);cv=cal.fit(oof.y,oof.p);dump(out/'calibrator.json',cv)
 m=e1.model(j['family'],study.best_params)
 with threadpool_limits(limits=6):m.fit(tr[e1.FEATURES],tr.y)
 joblib.dump(m,out/'model.joblib',compress=3);study.trials_dataframe().to_csv(out/'trials.csv',index=False)
 dump(out/'selection.json',dict(experiment=j['experiment'],family=j['family'],params=study.best_params,source_mean_AP=study.best_value,threshold=th,source_prior=tr.y.mean(),source_N=len(tr),source_keys_hash=j['source_keys_hash'],seed=e1.SEED,trials=len(study.trials),inner_fits=j['trials']*len(parts),best_trial_oof_reconstruction_fits=len(parts),final_fits=1,calibrator_fits=1,seconds=time.time()-start,target_not_scored_until_all_selections_locked=True))
 return j['experiment']+' '+j['family']+' source fit complete'
def metrics(y,p):
 pi=np.mean(y);ap=average_precision_score(y,p);return dict(N=len(y),positive=int(np.sum(y)),prevalence=pi,AP=ap,nAP=(ap-pi)/(1-pi),AUROC=roc_auc_score(y,p),Brier=np.mean((y-p)**2),LogLoss=log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1]))
def main():
 jobs=prep();print('E12 PROXY PLAN FROZEN',len(jobs),'source searches',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  for f in concurrent.futures.as_completed([pool.submit(worker,j) for j in jobs]):print(f.result(),flush=True)
 selected={};selections=[]
 for ex in sorted(set(j['experiment'] for j in jobs)):
  rr=[load(OUT/'proxy_fits'/ex/f/'selection.json') for f in FAM];best=max(rr,key=lambda z:(z['source_mean_AP'],-FAM.index(z['family'])));selected[ex]=best['family'];selections+=rr
 dump(OUT/'proxy_selection_lock.json',{'source_selected':selected,'all_source_studies_complete':True,'locked_before_target_scoring':True})
 pd.DataFrame(selections).to_csv(OUT/'E12_proxy_model_selection.csv',index=False)
 rows=[];paired={};pdest=OUT/'proxy_predictions';pdest.mkdir(exist_ok=True)
 for ex in selected:
  alltest=read(T/ex/'test.csv');keys=alltest.loc[alltest.domain.str.endswith('2014B'),'key'];d=read(T/ex/'data.csv').set_index('key',drop=False);te=d.loc[keys].sort_index()
  for family in FAM:
   out=OUT/'proxy_fits'/ex/family;m=joblib.load(out/'model.joblib');p=m.predict_proba(te[e1.FEATURES])[:,1].astype(float);q=cal.trans(p,load(out/'calibrator.json'));s=load(out/'selection.json');prior=np.full(len(te),s['source_prior']);z=te[['key','id','domain','y']].copy();z['raw']=p;z['calibrated']=q;z['source_prior']=prior;z.to_csv(pdest/f'{ex}_{family}_2014B.csv',index=False)
   assert abs(average_precision_score(te.y,p)-average_precision_score(te.y,q))<1e-12
   for method,score in [('raw',p),('calibrated',q),('source_prior',prior)]:rows.append(dict(experiment=ex,target_presentation='2014B',family=family,source_selected=family==selected[ex],method=method,analysis='proxy_end_plus_lag_0_30_60_same_source_set',**metrics(te.y.to_numpy(),score)))
  # Late-2014 proxy eligibility retains the complete original source; reuse that exact historical pipeline.
  old=read(A/'E6C_probability_transportability_v1.0'/('B_'+ex)/'target_predictions.csv');old=old[old.domain.str.endswith('2014J')].copy();old.to_csv(pdest/f'{ex}_archived_selected_2014J.csv',index=False)
  oldfamily=read(T/'source_selected_models.csv').set_index('experiment').loc[ex,'selected_model']
  for method in ['raw','calibrated','source_prior']:rows.append(dict(experiment=ex,target_presentation='2014J',family=oldfamily,source_selected=True,method=method,analysis='proxy_eligible_original_source_unchanged_archived_reuse',**metrics(old.y.to_numpy(),old[method].to_numpy())))
  selectedB=read(pdest/f'{ex}_{selected[ex]}_2014B.csv');combined=pd.concat([selectedB,old[selectedB.columns]],ignore_index=True).sort_values('key');combined.to_csv(pdest/f'{ex}_proxy_selected_combined.csv',index=False)
  for method in ['raw','calibrated','source_prior']:rows.append(dict(experiment=ex,target_presentation='2014B_and_2014J',family='source-selected per presentation policy',source_selected=True,method=method,analysis='combined_proxy_policy',**metrics(combined.y.to_numpy(),combined[method].to_numpy())))
  if ex!='Day28_withdrawal':paired[28 if ex=='Day28_failure' else 56]=combined
 pd.DataFrame(rows).to_csv(OUT/'E12_proxy_results.csv',index=False)
 pairdir=OUT/'new_paired_predictions';pairdir.mkdir(exist_ok=True)
 for pres in ['2014B','2014J','all']:
  a,b=[v if pres=='all' else v[v.domain.str.endswith(pres)] for v in [paired[28],paired[56]]];z=a.merge(b,on=['key','id'],suffixes=('_28','_56'),validate='one_to_one');assert np.array_equal(z.y_28,z.y_56);pd.DataFrame({'record_id':z.key,'id_student':z.id,'y':z.y_28,'pred_day28':z.raw_28,'pred_day56':z.raw_56}).sort_values('record_id').to_csv(pairdir/f'E12_proxy_selected_failure_{pres}.csv',index=False)
 dump(OUT/'proxy_execution_summary.json',dict(status='complete',new_trials=sum(j['trials'] for j in jobs),inner_fits=sum(j['trials']*len(j['folds']) for j in jobs),best_oof_reconstruction_fits=sum(len(j['folds']) for j in jobs),final_model_fits=len(jobs),new_calibrator_fits=len(jobs),calendar_verified=False,source_only=True))
 print('E12 PROXY COMPLETE',flush=True)
if __name__=='__main__':main()
