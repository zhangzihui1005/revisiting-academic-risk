"""E10B and E13: frozen source hyperparameters, independent fixed-policy fits."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='6'
from pathlib import Path
import json,hashlib,time,concurrent.futures,datetime
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];A=R/'archived_E1_E9/outputs';sys.path.insert(0,str(A/'experiment1_day28'));import run_experiment as e1
NEW=R/'experiments_new';FAM=['LR','XGBoost','CatBoost']
def read(p):return pd.read_csv(p,float_precision='round_trip')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def hashkeys(keys):return hashlib.sha256(('\n'.join(sorted(map(str,keys)))+'\n').encode()).hexdigest()
def objhash(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def prepare():
 jobs=[]
 e3=A/'E3_oulad_multihorizon_v1.0';t=A/'E4T_OULAD_temporal_generalization_v1.0'
 for env in ['internal','year2014_retrospective']:
  data={};roles={}
  for h in [28,56]:
   if env=='internal':
    d=read(e3/f'day{h}_features.csv').dropna(subset=['y_failure']);d['key']=d.code_module+'|'+d.code_presentation+'|'+d.id_student.astype(str);d['id']=d.id_student;d['y']=d.y_failure.astype(int);d['domain']=d.code_module+'_'+d.code_presentation
    roles[h]={s:set(d.loc[d.split.eq(s),'key']) for s in ['train','test']}
   else:
    ex='Day28_failure' if h==28 else 'E4T-B_day56_failure_secondary';d=read(t/ex/'data.csv');roles[h]={s:set(read(t/ex/(s+'.csv')).key) for s in ['train','test']}
   data[h]=d.set_index('key',drop=False)
  trkeys=sorted(roles[28]['train']&roles[56]['train']);tekeys=sorted(roles[28]['test']&roles[56]['test'])
  assert np.array_equal(data[28].loc[trkeys].y,data[56].loc[trkeys].y);assert np.array_equal(data[28].loc[tekeys].y,data[56].loc[tekeys].y)
  assert not set(data[28].loc[trkeys].id)&set(data[28].loc[tekeys].id)
  dest=NEW/'E10_temporal_control'/'fixed_inputs'/env;dest.mkdir(parents=True,exist_ok=True)
  for h in [28,56]:
   for role,keys in [('train',trkeys),('test',tekeys)]:data[h].loc[keys].to_csv(dest/f'{role}_day{h}.csv',index=False)
  for f in FAM:
   p=A/f'experiment1_day28/models/failure_{f}_parameters.json' if env=='internal' else t/f'Day28_failure/{f}/selection.json';cfg=load(p);params=cfg.get('params',cfg)
   for h in [28,56]:jobs.append(dict(id=f'E10B_{env}_{f}_D{h}',experiment='E10B',environment=env,family=f,horizon=h,features=e1.FEATURES,categorical=[],params=params,parameter_source=str(p.relative_to(R)),train=str(dest/f'train_day{h}.csv'),test=str(dest/f'test_day{h}.csv'),train_keys_hash=hashkeys(trkeys),test_keys_hash=hashkeys(tekeys),parameter_hash=objhash(params),feature_hash=objhash(e1.FEATURES),seed=e1.SEED,dest=str(NEW/'E10_temporal_control'/'fixed_fits'/f'{env}_{f}_D{h}')))
 for stage,root in [('internal',A/'E5A_UCI_information_horizons_v1.0'),('LOCO',A/'E5B_UCI_LOCO_v1.0')]:
  configs=load(root/'config.json');sel=read(root/'source_selected_models.csv').set_index('experiment')
  for ex,c in configs.items():
   if stage=='internal' and ex=='H0_expanded':continue
   feats=[x for x in c['features'] if x not in ['Unemployment rate','Inflation rate','GDP']];assert len(feats)==({'H0_strict':18,'H1':24,'H2':30}[ex] if stage=='internal' else 24)
   d=read(root/c['data']).set_index('row_id',drop=False);dest=NEW/'E13_UCI_availability'/'fixed_inputs'/stage/ex;dest.mkdir(parents=True,exist_ok=True)
   rr={}
   for role in ['train','test']:
    ids=read(root/ex/(role+'.csv')).row_id;part=d.loc[ids];part.to_csv(dest/(role+'.csv'),index=False);rr[role]=part
   assert not set(rr['train'].id)&set(rr['test'].id)
   if stage=='internal':assert len(rr['train'])==2178 and len(rr['test'])==726
   for f in (FAM if stage=='internal' else [sel.loc[ex,'selected_model']]):
    p=root/ex/f/'selection.json';params=load(p)['params'];jobs.append(dict(id=f'E13_{stage}_{ex}_{f}',experiment='E13',environment=stage,family=f,horizon=ex,features=feats,categorical=[x for x in c['categorical'] if x in feats],params=params,parameter_source=str(p.relative_to(R)),train=str(dest/'train.csv'),test=str(dest/'test.csv'),train_keys_hash=hashkeys(rr['train'].key),test_keys_hash=hashkeys(rr['test'].key),train_order_hash=objhash(rr['train'].key.tolist()),source_row_order='exact archived membership order',parameter_hash=objhash(params),feature_hash=objhash(feats),seed=e1.SEED,dest=str(NEW/'E13_UCI_availability'/'fixed_fits_v2'/f'{stage}_{ex}_{f}')))
 path=R/'revision_outputs/audit/fixed_controls_plan.json'
 plan={'specified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hardware_policy':'3 CPU fits x 6 threads; retain original CPU algorithms; bootstrap workers use remaining cores','no_new_search':True,'primary':'E10B internal LR; others sensitivity','E10B_year_interpretation':'original retrospective year separation, not verified calendar-asof','jobs':jobs}
 if path.exists() and load(path)['jobs']!=jobs:
  assert all(j['experiment']=='E10B' or 'source_row_order' in j for j in jobs)
  previous=path.with_name('fixed_controls_plan_initial_row_order.json')
  if not previous.exists():previous.write_bytes(path.read_bytes())
  dump(path.with_name('E13_row_order_correction.json'),{'reason':'Code audit found lexicographic key sorting added an unnecessary training-order change in minus-macro controls. Restore original archived train membership order. No target-based model/parameter selection.','initial_fits_retained':'E13_UCI_availability/fixed_fits','corrected_fits':'E13_UCI_availability/fixed_fits_v2','extra_model_fits':26,'changed_features_or_hyperparameters':False})
  dump(path,plan)
 elif not path.exists():dump(path,plan)
 return jobs
def worker(j):
 out=Path(j['dest']);out.mkdir(parents=True,exist_ok=True)
 if (out/'completed.json').exists():return j['id']+' cached'
 start=time.time();tr=read(j['train']);te=read(j['test']);assert hashkeys(tr.key)==j['train_keys_hash'] and hashkeys(te.key)==j['test_keys_hash'];assert not set(tr.id)&set(te.id)
 feats=j['features'];m=e1.model(j['family'],j['params'])
 if j['categorical']:
  cats=j['categorical'];pre=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),('numeric','passthrough',[x for x in feats if x not in cats])],sparse_threshold=0);m=make_pipeline(pre,m)
 with threadpool_limits(limits=6):m.fit(tr[feats],tr.y);p=m.predict_proba(te[feats])[:,1].astype(float)
 assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 joblib.dump(m,out/'model.joblib',compress=3)
 z=te[['key','id','domain','y']].rename(columns={'key':'record_id','id':'id_student'});z['p']=p;z.to_csv(out/'predictions.csv',index=False)
 pi=float(te.y.mean());ap=average_precision_score(te.y,p);order=np.lexsort((te.key.to_numpy(str),-p));k=int(np.ceil(.1*len(te)));prec=te.y.to_numpy()[order[:k]].mean()
 metrics=dict(job_id=j['id'],experiment=j['experiment'],environment=j['environment'],family=j['family'],horizon=j['horizon'],N=len(te),train_N=len(tr),positive=int(te.y.sum()),prevalence=pi,AP=ap,nAP=(ap-pi)/(1-pi),AUROC=roc_auc_score(te.y,p) if te.y.nunique()==2 else None,Brier=np.mean((te.y-p)**2),LogLoss=log_loss(te.y,np.clip(p,1e-12,1-1e-12),labels=[0,1]),Precision_at_10=prec,Recall_at_10=prec*k/te.y.sum(),Lift_at_10=prec/pi,feature_count=len(feats),train_keys_hash=j['train_keys_hash'],test_keys_hash=j['test_keys_hash'],parameter_hash=j['parameter_hash'],feature_hash=j['feature_hash'],seed=j['seed'],seconds=time.time()-start,prediction_file=str((out/'predictions.csv').relative_to(R)))
 dump(out/'configuration.json',j);dump(out/'completed.json',metrics);return j['id']+' completed '+str(round(metrics['seconds'],1))+'s'
def main():
 jobs=prepare();print('PLAN FROZEN',len(jobs),'fits',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  for f in concurrent.futures.as_completed([pool.submit(worker,j) for j in jobs]):print(f.result(),flush=True)
 results=[load(Path(j['dest'])/'completed.json') for j in jobs];d=pd.DataFrame(results)
 for exp,folder,name in [('E10B','E10_temporal_control','E10B_fixed_configuration_metrics.csv'),('E13','E13_UCI_availability','E13_fixed_configuration_metrics.csv')]:d[d.experiment.eq(exp)].to_csv(NEW/folder/name,index=False)
 print('ALL FIXED FITS COMPLETE',len(results),flush=True)
if __name__=='__main__':main()
