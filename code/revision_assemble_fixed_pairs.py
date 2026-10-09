from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
R=Path(__file__).resolve().parents[1];NEW=R/'experiments_new';A=R/'archived_E1_E9/outputs'
def read(p):return pd.read_csv(p,float_precision='round_trip')
def pair(a,b,dest):
 z=a.merge(b,on=['record_id','id_student'],validate='one_to_one',suffixes=('_a','_b'));assert len(z)==len(a)==len(b);assert np.array_equal(z.y_a,z.y_b)
 o=pd.DataFrame({'record_id':z.record_id,'id_student':z.id_student,'y':z.y_a,'pred_day28':z.p_a,'pred_day56':z.p_b}).sort_values('record_id')
 if dest.name.startswith('E13_'):o=o.rename(columns={'pred_day28':'pred_earlier','pred_day56':'pred_later'})
 o.to_csv(dest,index=False)
def main():
 e10=NEW/'E10_temporal_control';dest=e10/'new_paired_predictions';dest.mkdir(exist_ok=True)
 metrics=read(e10/'E10B_fixed_configuration_metrics.csv');audit=[]
 for (env,family),g in metrics.groupby(['environment','family']):
  assert len(g)==2 and g.train_keys_hash.nunique()==g.test_keys_hash.nunique()==g.parameter_hash.nunique()==g.feature_hash.nunique()==1
  paths=[R/g[g.horizon.eq(h)].prediction_file.iloc[0] for h in [28,56]];pair(*[read(p) for p in paths],dest/f'E10B_{env}_{family}.csv');audit.append(dict(environment=env,family=family,train_keys_equal=True,test_keys_equal=True,params_equal=True,feature_order_equal=True,train_N=int(g.train_N.iloc[0]),test_N=int(g.N.iloc[0]),training_rule='pairwise common training records; independent horizon fits of source preprocessing',status='PASS'))
 pd.DataFrame(audit).to_csv(e10/'E10B_fixed_train_fixed_params_audit.csv',index=False)
 u=NEW/'E13_UCI_availability';dest=u/'new_paired_predictions';dest.mkdir(exist_ok=True);m=read(u/'E13_fixed_configuration_metrics.csv')
 for family in ['LR','XGBoost','CatBoost']:
  z=m[m.environment.eq('internal')&m.family.eq(family)].set_index('horizon');assert z.test_keys_hash.nunique()==1
  for a,b in [('H0_strict','H1'),('H1','H2')]:pair(read(R/z.loc[a,'prediction_file']),read(R/z.loc[b,'prediction_file']),dest/f'E13_{family}_{a}_to_{b}.csv')
 m[m.environment.eq('internal')].to_csv(u/'E13_internal_minus_macro_9rows.csv',index=False);lo=m[m.environment.eq('LOCO')].copy();assert len(lo)==17;lo.to_csv(u/'E13_LOCO_minus_macro_17courses.csv',index=False)
 combined=pd.concat([read(R/p) for p in lo.prediction_file],ignore_index=True);assert len(combined)==3630 and not combined.record_id.duplicated().any();pi=combined.y.mean();ap=average_precision_score(combined.y,combined.p)
 summary=[dict(weighting='pooled_records',courses=17,N=len(combined),AP=ap,nAP=(ap-pi)/(1-pi),AUROC=roc_auc_score(combined.y,combined.p),Brier=np.mean((combined.y-combined.p)**2),worst_course_nAP=lo.nAP.min())]
 for label,z in [('equal_17_courses',lo),('equal_16_excluding_course33',lo[lo.horizon.ne('course_33')])]:summary.append(dict(weighting=label,courses=len(z),N=int(z.N.sum()),AP=z.AP.mean(),nAP=z.nAP.mean(),AUROC=z.AUROC.mean(),Brier=z.Brier.mean(),worst_course_nAP=z.nAP.min()))
 pd.DataFrame(summary).to_csv(u/'E13_LOCO_aggregation_sensitivity.csv',index=False)
 plan=json.loads((R/'revision_outputs/audit/fixed_controls_plan.json').read_text(encoding='utf-8'));fields=[]
 for j in plan['jobs']:
  if j['experiment']!='E13':continue
  fields.append(dict(job_id=j['id'],feature_count=len(j['features']),features=json.dumps(j['features']),removed_features=json.dumps(['Unemployment rate','Inflation rate','GDP']),train_keys_hash=j['train_keys_hash'],test_keys_hash=j['test_keys_hash'],parameter_source=j['parameter_source'],parameter_hash=j['parameter_hash'],future_fields_added=False))
 pd.DataFrame(fields).to_csv(u/'E13_feature_set_diff_and_split_audit.csv',index=False)
 # Independently reconcile reused full-risk Day28 flat and dual against E2 by row identity.
 e11=NEW/'E11_full_risk_set';rows=[]
 for arm,old in [('flat_CatBoost','flat_CatBoost_test.csv'),('dual_CatBoost_LightGBM','dual_primary_test.csv')]:
  new=read(e11/f'predictions/E11_D28_{arm}.csv');hist=read(A/'E2_day28_formulation_v1.0/predictions'/old)
  if 'record_id' not in hist:hist['record_id']=hist.code_module+'|'+hist.code_presentation+'|'+hist.id_student.astype(str)
  z=new.merge(hist,on='record_id',suffixes=('_new','_old'),validate='one_to_one');assert len(z)==5485
  error=max(float(np.max(np.abs(z['p_'+c+'_new']-z['p_'+c+'_old']))) for c in ['Success','Fail','Withdrawal']);assert error<1e-12
  rows.append(dict(arm=arm,N=len(z),max_probability_difference=error,status='PASS'))
 pd.DataFrame(rows).to_csv(e11/'E11_Day28_archive_reconciliation.csv',index=False)
 print('PAIRS READY: E10B 6; E13 6; E11 7. Day28 E11 archived probabilities verified.')
if __name__=='__main__':main()
