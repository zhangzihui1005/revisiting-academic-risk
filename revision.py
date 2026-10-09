"""Portable E10–E14 evidence checks and explicit fixed-configuration refits.
Verification reads saved predictions and draws; it does not sample or fit models.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 import pandas as pd
 return pd.read_csv(p,float_precision='round_trip')
def rel(s):return str(s).replace('\\','/')
def mapped(root,s):
 s=rel(s)
 if '/research_project/' in s:s=s.split('/research_project/',1)[1]
 p=(root/s).resolve();assert p.is_relative_to(root.resolve()),s
 return p
def restore(work):
 import reproduce
 reproduce.check()
 return reproduce.restore_payload(work/'revision/research_project','revision')
def close(a,b,label,tol=1e-10):
 import numpy as np
 assert np.allclose(a,b,atol=tol,rtol=0,equal_nan=True),(label,a,b)
def metric(y,p,keys):
 import numpy as np
 from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
 y=np.asarray(y);p=np.asarray(p);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 pi=y.mean();ap=average_precision_score(y,p);k=int(np.ceil(.1*len(y)));ix=np.lexsort((np.asarray(keys).astype(str),-p));prec=y[ix[:k]].mean()
 return dict(N=len(y),positive=int(y.sum()),prevalence=pi,AP=ap,nAP=(ap-pi)/(1-pi),AUROC=roc_auc_score(y,p) if len(np.unique(y))==2 else float('nan'),Brier=np.mean((y-p)**2),LogLoss=log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1]),Precision_at_10=prec,Recall_at_10=prec*k/y.sum(),Lift_at_10=prec/pi)
def check_metrics(y,p,keys,row):
 m=metric(y,p,keys)
 for k,v in m.items():
  if k in row:close(v,row[k],k)
 return m
def jobs(root):return load(root/'revision_outputs/audit/fixed_controls_plan.json')['jobs']
def preflight(root):
 import numpy as np
 rows=[]
 def keyhash(k):return hashlib.sha256(('\n'.join(sorted(map(str,k)))+'\n').encode()).hexdigest()
 def objhash(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 for j in jobs(root):
  tr=read(mapped(root,j['train']));te=read(mapped(root,j['test']))
  assert keyhash(tr.key)==j['train_keys_hash'] and keyhash(te.key)==j['test_keys_hash'],j['id']
  assert not set(tr.id)&set(te.id),j['id']
  assert objhash(j['params'])==j['parameter_hash'] and objhash(j['features'])==j['feature_hash']
  assert set(j['features'])<=set(tr.columns)&set(te.columns)
  assert j['seed']==20261007
  if j['experiment']=='E13':
   assert objhash(tr.key.tolist())==j['train_order_hash']
   assert not set(['Unemployment rate','Inflation rate','GDP'])&set(j['features'])
  rows.append(dict(job=j['id'],training_records=len(tr),test_records=len(te),features=len(j['features']),status='passed'))
 assert len(rows)==38
 return rows
def derived(root,table,dest):
 """S32 reconstruction: subtract synchronized SAVED draws, never resample."""
 import numpy as np,pandas as pd
 pairs=[('E11 flat failure gain minus withdrawal gain','E11_flat_CatBoost_Fail','E11_flat_CatBoost_Withdrawal',.0623,.0234,.1002)]
 for f,expected in [('LR',(.2648,.1909,.3339)),('XGBoost',(.2887,.2214,.3587)),('CatBoost',(.2658,.1910,.3422))]:pairs.append((f'E13 {f} first increment minus second increment',f'E13_{f}_H0_strict_to_H1',f'E13_{f}_H1_to_H2',*expected))
 rows=[];table=table.set_index('analysis')
 for name,a,b,*expected in pairs:
  x,y=table.loc[a],table.loc[b];px=mapped(root,x.replicates_file);py=mapped(root,y.replicates_file)
  assert (px.parent/'cluster_order.csv').read_bytes()==(py.parent/'cluster_order.csv').read_bytes()
  rx=load(px.parent/'rng_state.json');ry=load(py.parent/'rng_state.json')
  for k in ['initial','final','generator','numpy_version','cluster_order_sha256']:assert rx[k]==ry[k],(name,k)
  dx,dy=read(px),read(py);assert dx.draw_id.equals(dy.draw_id) and dx.valid.equals(dy.valid)
  # Same record/cluster assignment; labels may differ for Fail versus Withdrawal.
  ix=read(mapped(root,x.input));iy=read(mapped(root,y.input));assert ix[['record_id','id_student']].equals(iy[['record_id','id_student']])
  dif=dx.loc[dx.valid,'delta_nAP']-dy.loc[dy.valid,'delta_nAP'];lo,hi=dif.quantile([.025,.975]);point=x.delta_nAP-y.delta_nAP
  close([point,lo,hi],expected,name,tol=.0000500001)
  rows.append(dict(comparison=name,delta_nAP_difference=point,CI95_low=lo,CI95_high=hi,saved_draws=len(dif),synchronization='cluster order, row mapping, draw IDs, valid flags and RNG states match',classification='post-hoc descriptive; no multiplicity adjustment',first=a,second=b))
 pd.DataFrame(rows).to_csv(dest/'S32_verified_posthoc_contrasts.csv',index=False)
 return rows
def verify(root,work):
 import numpy as np,pandas as pd
 from sklearn.metrics import f1_score,log_loss
 work.mkdir(parents=True,exist_ok=True);N=root/'experiments_new';T=root/'revision_outputs/tables'
 tab=read(T/'S26_E10_E14_paired_horizon_comparisons.csv');paired=[]
 for _,r in tab.iterrows():
  p=mapped(root,r.input);assert sha(p)==r.input_sha256
  d=read(p);assert not d.record_id.duplicated().any();assert len(d)==r.N and d.id_student.nunique()==r.unique_students
  assert int(d.y.sum())==r.positive
  a,b=('pred_earlier','pred_later') if 'pred_earlier' in d else ('pred_day28','pred_day56')
  x,y=metric(d.y,d[a],d.record_id),metric(d.y,d[b],d.record_id)
  close([x['AP'],y['AP'],x['nAP'],y['nAP'],y['nAP']-x['nAP']],[r.AP_earlier,r.AP_later,r.nAP_earlier,r.nAP_later,r.delta_nAP],r.analysis)
  reps=read(mapped(root,r.replicates_file));good=reps.loc[reps.valid,'delta_nAP'];assert len(reps)==r.draws_attempted and len(good)==r.draws_valid
  close(good.quantile([.025,.975]),[r.CI95_low,r.CI95_high],r.analysis)
  paired.append(dict(analysis=r.analysis,N=len(d),delta_nAP=y['nAP']-x['nAP'],existing_draws=len(good),status='passed'))
 assert len(paired)==79
 fixed=[]
 for file in ['S27_E10B_fixed_configuration_metrics.csv','S30_E13_minus_macro_metrics.csv']:
  for _,r in read(T/file).iterrows():
   d=read(mapped(root,r.prediction_file));check_metrics(d.y,d.p,d.record_id,r);fixed.append(r.job_id)
 assert len(fixed)==38
 # UCI weighting sensitivity: course 33 remains in the primary mean.
 uci=read(T/'S30_E13_minus_macro_metrics.csv');lo=uci[uci.environment.eq('LOCO')]
 allpred=pd.concat([read(mapped(root,p)) for p in lo.prediction_file],ignore_index=True)
 assert len(lo)==17 and len(allpred)==3630 and not allpred.record_id.duplicated().any()
 agg=read(T/'S30b_E13_LOCO_aggregation.csv').set_index('weighting')
 pooled=metric(allpred.y,allpred.p,allpred.record_id)
 for k in ['AP','nAP','AUROC','Brier']:close(pooled[k],agg.loc['pooled_records',k],'UCI pooled '+k)
 for name,sub in [('equal_17_courses',lo),('equal_16_excluding_course33',lo[lo.horizon.ne('course_33')])]:
  for k in ['AP','nAP','AUROC','Brier']:close(sub[k].mean(),agg.loc[name,k],name+' '+k)
 # E11 dynamic scoring: all eligible active enrollments, no outcome filtering.
 e11=N/'E11_full_risk_set';binary=read(e11/'E11_fullrisk_internal_dynamic.csv')
 for _,r in binary.iterrows():
  d=read(e11/f'predictions/E11_D{r.horizon}_{r.arm}.csv');label=1 if r.task=='Fail' else 2
  check_metrics(d.y_three.eq(label).astype(int),d['p_'+r.task],d.record_id,r)
 multi=read(e11/'E11_multiclass_metrics.csv');conf=read(e11/'E11_confusion_matrices.csv');classes=['Success','Fail','Withdrawal']
 for _,r in multi.iterrows():
  d=read(e11/f'predictions/E11_D{r.horizon}_{r.arm}.csv');p=d[['p_'+c for c in classes]].to_numpy();y=d.y_three.to_numpy(int);pred=p.argmax(axis=1)
  close(p.sum(axis=1),1,'three-class probability sum');close(np.mean(np.sum((np.eye(3)[y]-p)**2,axis=1)),r.MulticlassBrier,'multiclass Brier');close(f1_score(y,pred,average='macro'),r.MacroF1,'macro F1');close(log_loss(y,p,labels=[0,1,2]),r.MulticlassLogLoss,'multiclass logloss')
  z=conf[conf.horizon.eq(r.horizon)&conf.arm.eq(r.arm)]
  for _,c in z.iterrows():assert int(((y==classes.index(c.true_class))&(pred==classes.index(c.predicted_class))).sum())==c.N
 # E12 all saved proxy prediction vectors, including presentation-specific reuse.
 e12=N/'E12_calendar_audit';proxy=read(e12/'E12_proxy_results.csv');proxy_checked=0
 for _,r in proxy.iterrows():
  if r.target_presentation=='2014B':name=f'{r.experiment}_{r.family}_2014B.csv'
  elif r.target_presentation=='2014J':name=f'{r.experiment}_archived_selected_2014J.csv'
  else:name=f'{r.experiment}_proxy_selected_combined.csv'
  p=e12/'proxy_predictions'/name
  assert p.exists(),name
  d=read(p);column={'raw':'raw','Platt':'calibrated','platt':'calibrated','source_prior':'source_prior'}.get(r.method,r.method)
  check_metrics(d.y,d[column],d.key,r);proxy_checked+=1
 # E14 equal-module and pooled loss are distinct estimands.
 e14=N/'E14_calibration_baseline';d=read(e14/'E14_matched_predictions.csv');assert len(d)==27421 and d.id_student.nunique()==24749
 assert not d.record_id.duplicated().any() and d.code_module.nunique()==7
 observed={}
 for m in ['raw','platt','prior']:
  p=d['p_'+m].to_numpy();y=d.y.to_numpy();q=np.clip(p,1e-12,1-1e-12)
  for k,loss in [('Brier',(y-p)**2),('LogLoss',-(y*np.log(q)+(1-y)*np.log1p(-q)))]:
   observed[f'equal7_{m}_{k}']=pd.Series(loss).groupby(d.code_module).mean().mean();observed[f'pooled_{m}_{k}']=loss.mean()
 for wt in ['equal7','pooled']:
  for met in ['Brier','LogLoss']:
   for base in ['prior','raw']:observed[f'{wt}_delta_platt_minus_{base}_{met}']=observed[f'{wt}_platt_{met}']-observed[f'{wt}_{base}_{met}']
 summary=load(e14/'bootstrap/summary.json');draws=read(e14/'bootstrap/E14_equal_module_global_student_bootstrap.csv');assert len(draws)==5000
 for k,v in observed.items():close(v,summary['observed'][k],k)
 for r in summary['contrasts']:close([observed[r['contrast']],*draws[r['contrast']].quantile([.025,.975])],[r['estimate'],r['CI95_low'],r['CI95_high']],r['contrast'])
 posthoc=derived(root,tab,work)
 # Recovered historical E8 intervals: reduced-minus-full -> full-minus-reduced.
 abl=read(root/'archived_support/E8/fixed_horizon_ablation.csv').copy()
 abl['full_minus_reduced_nAP']=-abl.delta_nAP;abl['loss_CI95_low']=-abl.delta_nAP_CI_high;abl['loss_CI95_high']=-abl.delta_nAP_CI_low
 abl.to_csv(work/'S23_historical_ablation_loss.csv',index=False)
 pf=preflight(root)
 pd.DataFrame(paired).to_csv(work/'verified_revision_contrasts.csv',index=False)
 pd.DataFrame(pf).to_csv(work/'verified_fixed_input_audit.csv',index=False)
 report=dict(status='passed',horizon_contrasts=len(paired),existing_horizon_draws=sum(x['existing_draws'] for x in paired),fixed_configuration_metric_rows=len(fixed),fixed_input_audits=len(pf),E13_aggregation_summaries=3,E11_dynamic_rows=len(binary),E11_multiclass_rows=len(multi),E11_confusion_cells=len(conf),E12_proxy_rows=proxy_checked,E14_global_cluster_draws=len(draws),E14_contrasts=len(summary['contrasts']),S32_saved_draw_contrasts=len(posthoc),S23_historical_ablation_rows=len(abl),fits=0,new_bootstrap_draws=0,scope='Saved predictions, saved interval quantiles, input/configuration audits; no new model selection or statistical sampling.')
 dump(work/'REVISION_VERIFICATION.json',report);print(json.dumps(report),flush=True)
 return report
def replay(root,work,job_id):
 """Refit one explicitly selected archived E10B/E13 configuration, no search."""
 import numpy as np,pandas as pd
 from sklearn.compose import ColumnTransformer
 from sklearn.preprocessing import OneHotEncoder
 from sklearn.pipeline import make_pipeline
 from threadpoolctl import threadpool_limits
 preflight(root);matches=[j for j in jobs(root) if j['id']==job_id]
 if len(matches)!=1:raise SystemExit('Unknown job. Run: python revision.py list-fixed')
 j=matches[0];dest=work/'fixed_replays'/job_id
 if dest.exists():raise SystemExit('Replay output exists; use a fresh --workdir.')
 dest.mkdir(parents=True)
 spec=importlib.util.spec_from_file_location('original_e1',HERE/'code/e1_model_and_feature_engine.py');e1=importlib.util.module_from_spec(spec);spec.loader.exec_module(e1)
 assert e1.SEED==j['seed'];tr=read(mapped(root,j['train']));te=read(mapped(root,j['test']));m=e1.model(j['family'],j['params']);feats=j['features']
 if j['categorical']:
  cats=j['categorical'];pre=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),('numeric','passthrough',[x for x in feats if x not in cats])],sparse_threshold=0);m=make_pipeline(pre,m)
 with threadpool_limits(limits=6):m.fit(tr[feats],tr.y);p=m.predict_proba(te[feats])[:,1].astype(float)
 old=read(mapped(root,j['dest'])/'predictions.csv');assert old.record_id.astype(str).tolist()==te.key.astype(str).tolist()
 delta=float(np.max(abs(p-old.p)));ok=delta<1e-7
 pd.DataFrame({'record_id':te.key,'id_student':te.id,'y':te.y,'p':p}).to_csv(dest/'predictions.csv',index=False)
 portable=dict(j);portable['train']=mapped(root,j['train']).relative_to(root).as_posix();portable['test']=mapped(root,j['test']).relative_to(root).as_posix();portable['dest']='fixed_replays/'+job_id
 dump(dest/'configuration.json',portable);dump(dest/'comparison.json',dict(status='passed' if ok else 'numerical_difference',job=job_id,max_prediction_difference=delta,tolerance=1e-7,new_fits=1,new_search_trials=0,new_bootstrap_draws=0))
 print(job_id,'max prediction difference',delta,flush=True)
 if not ok:raise SystemExit('Prediction tolerance exceeded; inspect comparison.json and dependency/platform versions.')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['verify','preflight','list-fixed','replay-fixed']);p.add_argument('--workdir',type=Path,default=HERE/'_work');p.add_argument('--job');a=p.parse_args();work=a.workdir.resolve();root=restore(work)
 if a.action=='verify':verify(root,work)
 elif a.action=='preflight':dump(work/'REVISION_PREFLIGHT.json',dict(status='passed',jobs=preflight(root),fits=0))
 elif a.action=='list-fixed':
  for j in jobs(root):print(j['id'])
 elif not a.job:raise SystemExit('Specify --job from list-fixed.')
 else:replay(root,work,a.job)
if __name__=='__main__':main()
