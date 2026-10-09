import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
SEED=20261007;NAMES=['LR','XGBoost','CatBoost']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def specs(parent,extra_temporal=False):
 a=parent/'E4A_day28_student_disjoint_LOMO_v1.0';t=parent/'E4T_OULAD_temporal_generalization_v1.0';u=parent/'E5B_UCI_LOCO_v1.0';out=[]
 ca=pd.read_csv(a/'inner_selected_models.csv')
 for s in ca.itertuples():out.append(dict(key='A_'+s.task+'_'+s.heldout_module,environment='A',setting='A_'+s.task,task=s.task,outer=s.heldout_module,base=str(a/'folds'/s.heldout_module/s.task),old=True,selected=s.inner_selected_model,source_root=a.name))
 ct=pd.read_csv(t/'source_selected_models.csv').set_index('experiment');exps=['Day28_failure'] if not extra_temporal else list(ct.index)
 for ex in exps:out.append(dict(key='B_'+ex,environment='B',setting='B_'+ex,task='withdrawal' if 'withdrawal' in ex else 'failure',outer='2014',base=str(t/ex),old=False,selected=ct.loc[ex,'selected_model'],source_root=t.name))
 cu=pd.read_csv(u/'source_selected_models.csv')
 for s in cu.itertuples():out.append(dict(key='C_'+s.experiment,environment='C',setting='C_UCI_H1',task='dropout',outer=s.experiment,base=str(u/s.experiment),old=False,selected=s.selected_model,source_root=u.name))
 return out
def aligned(s,role):
 out=None
 for n in NAMES:
  p=Path(s['base'])/n/('best_inner_predictions.csv.gz' if role=='source' else 'test_predictions.csv');z=pd.read_csv(p,float_precision='round_trip')
  if s['old']:
   z=z.rename(columns={'enrolment_key':'key','id_student':'id'});z['domain']=z.code_module;z['module']=z.code_module
   if n=='XGBoost':z['probability']=z.probability.astype(np.float32).astype(float)
  z['domain']=z.domain.astype(str);z['module']=z.module.astype(str);z=z.sort_values('key').reset_index(drop=True)
  if out is None:out=z[['key','row_id','id','domain','module','y']].copy()
  else:
   for c in ['key','row_id','id','domain','y']:assert np.array_equal(out[c],z[c]),(s['key'],n,c)
  out[n]=z.probability.to_numpy(dtype=float)
 assert not out.key.duplicated().any();assert np.isfinite(out[NAMES]).all().all();return out
def metrics(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pi=y.mean();ap=average_precision_score(y,p)
 return dict(N=len(y),positive=int(y.sum()),prevalence=pi,AP=ap,nAP=(ap-pi)/(1-pi) if 0<pi<1 else np.nan,AUROC=roc_auc_score(y,p) if 0<pi<1 else np.nan,Brier=brier_score_loss(y,p),LogLoss=log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1]))
def domain_rows(z,methods,meta):
 return [dict(**meta,domain=str(g),method=n,**metrics(v.y,v[n])) for g,v in z.groupby('domain',sort=True) for n in methods]
def summarize(rows):
 d=pd.DataFrame(rows);out=[]
 for (setting,n),z in d.groupby(['setting','method']):
  r=dict(setting=setting,method=n,domain_cells=len(z))
  for m in ['AP','nAP','AUROC','Brier','LogLoss']:r.update({f'Mean_{m}':z[m].mean(),f'SD_{m}':z[m].std(ddof=0),f'Min_{m}':z[m].min(),f'Max_{m}':z[m].max()})
  out.append(r)
 return pd.DataFrame(out)
