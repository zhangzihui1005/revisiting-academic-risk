"""Saved paired student bootstrap, optimized AP with exact tie handling."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,argparse,concurrent.futures,time
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
R=Path(__file__).resolve().parents[1];E10=R/'experiments_new/E10_temporal_control';E14=R/'experiments_new/E14_calibration_baseline'
def read(p):return pd.read_csv(p,float_precision='round_trip')
def dump(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class WeightedAP:
 def __init__(self,y,p):
  self.order=np.argsort(-np.asarray(p),kind='stable');self.y=np.asarray(y)[self.order];s=np.asarray(p)[self.order];self.ends=np.r_[np.where(s[:-1]!=s[1:])[0],len(s)-1]
 def __call__(self,w):
  w=w[self.order];tp=np.cumsum(w*self.y)[self.ends];cw=np.cumsum(w)[self.ends]
  pos=tp[-1];total=cw[-1]
  if pos<=0 or pos>=total:return np.nan,np.nan
  ap=np.sum(np.diff(np.r_[0,tp])*np.divide(tp,cw,out=np.zeros_like(tp,dtype=float),where=cw>0))/pos;pi=pos/total
  return float(ap),float((ap-pi)/(1-pi))
def validate_engine():
 rng=np.random.default_rng(887);maxerr=0
 for n in [10,100,1000]:
  for _ in range(20):
   y=rng.integers(0,2,n);p=np.round(rng.random(n),1);w=rng.integers(0,4,n)
   if np.dot(w,y)==0 or np.dot(w,1-y)==0:continue
   ap,_=WeightedAP(y,p)(w);ref=average_precision_score(y,p,sample_weight=w);maxerr=max(maxerr,abs(ap-ref));assert abs(ap-ref)<1e-12
 dump(R/'revision_outputs/audit/bootstrap_engine_validation.json',{'synthetic_only':True,'weighted_AP_tie_and_zero_weight_checks':60,'max_absolute_error_vs_sklearn':maxerr,'note':'Functional tests are not scientific observations.'})
def bootstrap_pair(input_path,outdir,draws=2000,seed=20261009):
 p=Path(input_path);out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
 if (out/'summary.json').exists():
  cached=json.loads((out/'summary.json').read_text(encoding='utf-8'));assert cached['input_sha256']==sha(p),'Changed bootstrap input: '+str(p);return cached
 d=read(p)
 if 'pred_earlier' in d:d=d.rename(columns={'pred_earlier':'pred_day28','pred_later':'pred_day56'})
 assert not d.record_id.duplicated().any();assert not d[['y','id_student','pred_day28','pred_day56']].isna().any().any()
 groups,inv=np.unique(d.id_student,return_inverse=True);n=len(groups);rng=np.random.default_rng(seed);initial=rng.bit_generator.state
 pd.DataFrame({'cluster_index':range(n),'id_student':groups}).to_csv(out/'cluster_order.csv',index=False)
 f=WeightedAP(d.y,d.pred_day28);g=WeightedAP(d.y,d.pred_day56);w=np.ones(len(d));obs1=f(w);obs2=g(w);rows=[]
 for i in range(draws):
  counts=np.bincount(rng.integers(0,n,size=n),minlength=n);w=counts[inv];a=f(w);b=g(w);valid=np.isfinite(a[1]) and np.isfinite(b[1]);rows.append((i,valid,a[0],b[0],a[1],b[1],b[1]-a[1],np.dot(w,d.y)/w.sum()))
 vals=pd.DataFrame(rows,columns=['draw_id','valid','AP_earlier','AP_later','nAP_earlier','nAP_later','delta_nAP','resampled_prevalence']);vals.to_csv(out/'replicates.csv',index=False)
 ok=vals.loc[vals.valid,'delta_nAP'];ci=ok.quantile([.025,.975]).to_numpy()
 summary=dict(analysis=p.stem,input=str(p.relative_to(R)),input_sha256=sha(p),N=len(d),unique_students=n,positive=int(d.y.sum()),prevalence=d.y.mean(),AP_earlier=obs1[0],AP_later=obs2[0],nAP_earlier=obs1[1],nAP_later=obs2[1],delta_nAP=obs2[1]-obs1[1],CI95_low=ci[0],CI95_high=ci[1],draws_attempted=draws,draws_valid=len(ok),seed=seed,bootstrap_unit='global student ID' if not p.stem.startswith('E13') else 'released UCI record identity',delta_direction='later minus earlier',inference='fixed fitted models and records; revision-stage descriptive percentile CI; no multiplicity control',replicates_file=str((out/'replicates.csv').relative_to(R)))
 dump(out/'rng_state.json',{'initial':initial,'final':rng.bit_generator.state,'generator':'numpy.random.PCG64','numpy_version':np.__version__,'draw_algorithm':'rng.integers(0,n_clusters,size=n_clusters), then bincount, repeat in draw_id order','cluster_order_sha256':sha(out/'cluster_order.csv')})
 dump(out/'summary.json',summary);return summary
def e14():
 out=E14/'bootstrap';out.mkdir(exist_ok=True);p=E14/'E14_matched_predictions.csv';d=read(p)
 groups,inv=np.unique(d.id_student,return_inverse=True);domains,di=np.unique(d.code_module,return_inverse=True);assert len(domains)==7
 pd.DataFrame({'cluster_index':range(len(groups)),'id_student':groups}).to_csv(out/'cluster_order.csv',index=False)
 y=d.y.to_numpy();losses={}
 for m in ['raw','platt','prior']:
  q=d['p_'+m].to_numpy();losses[(m,'Brier')]=(y-q)**2;q=np.clip(q,1e-12,1-1e-12);losses[(m,'LogLoss')]=-(y*np.log(q)+(1-y)*np.log1p(-q))
 def evalw(w):
  den=np.bincount(di,weights=w,minlength=7);result={}
  for (m,metric),loss in losses.items():
   num=np.bincount(di,weights=w*loss,minlength=7);result[f'equal7_{m}_{metric}']=np.mean(num/den) if (den>0).all() else np.nan;result[f'pooled_{m}_{metric}']=num.sum()/den.sum()
  for wt in ['equal7','pooled']:
   for metric in ['Brier','LogLoss']:
    for base in ['prior','raw']:result[f'{wt}_delta_platt_minus_{base}_{metric}']=result[f'{wt}_platt_{metric}']-result[f'{wt}_{base}_{metric}']
  return result
 obs=evalw(np.ones(len(d)));rng=np.random.default_rng(20261009);initial=rng.bit_generator.state;rows=[];n=len(groups)
 for i in range(5000):
  w=np.bincount(rng.integers(0,n,size=n),minlength=n)[inv];row=evalw(w);row['draw_id']=i;rows.append(row)
 vals=pd.DataFrame(rows);vals.to_csv(out/'E14_equal_module_global_student_bootstrap.csv',index=False);summary=[]
 for name,value in obs.items():
  if 'delta_' not in name:continue
  good=vals[name].dropna();lo,hi=good.quantile([.025,.975]);summary.append(dict(contrast=name,estimate=value,CI95_low=lo,CI95_high=hi,draws_attempted=5000,draws_valid=len(good),seed=20261009,unit='global student ID',N=len(d),students=n))
 pd.DataFrame(summary).to_csv(E14/'E14_paired_comparisons.csv',index=False);dump(out/'rng_state.json',{'initial':initial,'final':rng.bit_generator.state,'input_sha256':sha(p),'numpy_version':np.__version__,'cluster_order_sha256':sha(out/'cluster_order.csv'),'draw_algorithm':'PCG64 rng.integers(0,n,size=n); global clusters; fixed equal-module weighting'})
 dump(out/'summary.json',{'observed':obs,'contrasts':summary,'inference':'conditional on fixed models and records, not equivalence or deployment evidence'})
 return 'E14 5000 global-cluster draws completed'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['initial','fixed'],default='initial');args=ap.parse_args();validate_engine()
 jobs=[]
 if args.mode=='initial':
  for p in sorted((E10/'paired_predictions').glob('*.csv')):jobs.append((p,E10/'bootstrap_E10A'/p.stem))
 else:
  for folder in ['E10_temporal_control','E13_UCI_availability','E11_full_risk_set','E12_calendar_audit']:
   root=R/'experiments_new'/folder
   for p in sorted((root/'new_paired_predictions').glob('*.csv')):jobs.append((p,root/'bootstrap_new'/p.stem))
 print('BOOTSTRAP JOBS',len(jobs),flush=True);results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2 if args.mode=='initial' else 16) as pool:
  futures={pool.submit(bootstrap_pair,*j):str(j[0]) for j in jobs}
  if args.mode=='initial':futures[pool.submit(e14)]='E14'
  for f in concurrent.futures.as_completed(futures):
   result=f.result()
   if isinstance(result,dict):results.append(result);print(result['analysis'],'done',flush=True)
   else:print(result,flush=True)
 pd.DataFrame(results).to_csv(R/'revision_outputs'/('paired_bootstrap_'+args.mode+'_summary.csv'),index=False)
if __name__=='__main__':main()
