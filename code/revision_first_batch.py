"""E10--E14 initial gates: frozen predictions only; no training or new bootstrap."""
import sys,os
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,hashlib,calendar,datetime,zipfile
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'research_project'; A=R/'archived_E1_E9'/'outputs'
E10=R/'experiments_new/E10_temporal_control';E12=R/'experiments_new/E12_calendar_audit';E14=R/'experiments_new/E14_calibration_baseline';AUD=R/'revision_outputs/audit'
E3=A/'E3_oulad_multihorizon_v1.0';T=A/'E4T_OULAD_temporal_generalization_v1.0';L28=A/'E4A_day28_student_disjoint_LOMO_v1.0';L56=A/'E4B_day56_LOMO_secondary_v1.0';C=A/'E6C_probability_transportability_v1.0'
FAMILIES=['LR','XGBoost','CatBoost'];MODULES=list('ABCDEFG')
def read(p):return pd.read_csv(p,float_precision='round_trip')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def keyhash(keys):return hashlib.sha256(('\n'.join(sorted(map(str,keys)))+'\n').encode()).hexdigest()
def ohash(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def norm(d,task,score):
 d=d.copy().rename(columns={'enrolment_key':'record_id','key':'record_id','id':'id_student','module':'code_module'})
 if 'record_id' not in d:d['record_id']=d.code_module+'|'+d.code_presentation+'|'+d.id_student.astype(str)
 if 'code_module' not in d:d['code_module']=d.record_id.str.split('|').str[0]
 if 'code_presentation' not in d:d['code_presentation']=d.record_id.str.split('|').str[1]
 if 'y' not in d:d['y']=d['y_'+task]
 d['y']=d.y.astype(int);d['p']=d[score].astype(float)
 # Historical E1/E3/LOMO XGB files used short float32 text; recover native scores.
 if score=='XGBoost_probability':d['p']=d.p.astype(np.float32).astype(float)
 assert not d.record_id.duplicated().any();assert d.y.isin([0,1]).all();assert np.isfinite(d.p).all() and d.p.between(0,1).all()
 return d[['record_id','id_student','code_module','code_presentation','y','p']].sort_values('record_id').reset_index(drop=True)
def metrics(d):
 pi=float(d.y.mean());ap=float(average_precision_score(d.y,d.p))
 return {'N':len(d),'students':d.id_student.nunique(),'positives':int(d.y.sum()),'prevalence':pi,'AP':ap,'nAP':(ap-pi)/(1-pi),'AUROC':roc_auc_score(d.y,d.p)}
def pair(a,b):
 z=a.merge(b,on=['record_id','id_student','code_module','code_presentation'],suffixes=('_28','_56'),validate='one_to_one')
 assert len(z)>0 and (z.y_28==z.y_56).all()
 return z.rename(columns={'y_28':'y','p_28':'pred_day28','p_56':'pred_day56'}).drop(columns='y_56').sort_values('record_id').reset_index(drop=True)
CHECKS=[]
def check(name,got,expected,tol=1e-12):
 ok=abs(float(got)-float(expected))<=tol
 CHECKS.append(dict(check=name,recomputed=got,expected=expected,tolerance=tol,status='PASS' if ok else 'FAIL'))
 if not ok:pd.DataFrame(CHECKS).to_csv(AUD/'baseline_reconciliation.csv',index=False);raise AssertionError(CHECKS[-1])
def baseline():
 for task in ['failure','withdrawal']:
  ds=[]
  for h in [28,56]:
   family=load(E3/f'day{h}/{task}_selection.json')['model'];ds.append(norm(read(E3/f'day{h}/{task}_stable_test_predictions.csv'),task,family+'_probability'))
  z=pair(*ds);check('E3_four_landmark_'+task+'_N',len(z),4474 if task=='failure' else 5105)
  delta=metrics(ds[1])['nAP']-metrics(ds[0])['nAP'];orig=read(E3/'stable_paired_horizon_gains.csv').query('task==@task and from_day==28').iloc[0]
  check('E3_selected_'+task+'_delta_nAP',delta,orig.delta_nAP)
 a=norm(read(T/'Day28_failure/LR/test_predictions.csv'),'failure','probability');b=norm(read(T/'E4T-B_day56_failure_secondary/LR/test_predictions.csv'),'failure','probability');z=pair(a,b)
 check('E4T_common_failure_N',len(z),12616)
 ma=metrics(z.rename(columns={'pred_day28':'p'}));mb=metrics(z.rename(columns={'pred_day56':'p'}))
 orig=read(T/'paired_horizon_comparison.csv').query("scope=='all' and comparison=='Validation-selected'").iloc[0]
 check('E4T_selected_failure_delta_nAP',mb['nAP']-ma['nAP'],orig.delta_nAP)
 frames=[];rows=[]
 for m in ['AAA','BBB','CCC','DDD','EEE','FFF','GGG']:
  p=C/f'A_withdrawal_{m}/target_predictions.csv';d=read(p);tr=read(L28/f'folds/{m}/withdrawal/train_membership.csv')
  prior=float(tr.y_withdrawal.mean());assert not set(tr.id_student)&set(d.id)
  check('E14_'+m+'_source_prior',d.source_prior.iloc[0],prior)
  assert np.allclose(d.source_prior,prior,atol=1e-15,rtol=0)
  raw=read(L28/f'folds/{m}/withdrawal/LR/test_predictions.csv').set_index('enrolment_key');aligned=raw.loc[d.key]
  assert np.array_equal(aligned.y,d.y);assert np.allclose(aligned.probability,d.raw,atol=1e-15,rtol=0)
  rows.append(dict(module=m,N=len(d),positives=int(d.y.sum()),source_N=len(tr),source_prior=prior,raw_Brier=np.mean((d.y-d.raw)**2),platt_Brier=np.mean((d.y-d.calibrated)**2),prior_Brier=np.mean((d.y-prior)**2),source_target_student_overlap=0))
  d=d.rename(columns={'key':'record_id','id':'id_student','module':'code_module','raw':'p_raw','calibrated':'p_platt','source_prior':'p_prior'})
  frames.append(d[['record_id','id_student','code_module','y','p_raw','p_platt','p_prior']])
 x=pd.DataFrame(rows);s=read(C/'calibration_summary.csv').query("setting=='A_withdrawal'").set_index('method')
 check('E6C_equal7_raw_Brier',x.raw_Brier.mean(),s.loc['raw','Mean_Brier']);check('E6C_equal7_platt_Brier',x.platt_Brier.mean(),s.loc['calibrated','Mean_Brier'])
 prior_expected=read(L28/'probability_quality_diagnostics.csv').query("task=='withdrawal'").source_prior_constant_Brier.mean();check('E4A_equal7_prior_Brier',x.prior_Brier.mean(),prior_expected)
 x.to_csv(E14/'E14_cal_vs_prior_module_reconciled.csv',index=False);pd.concat(frames).sort_values('record_id').to_csv(E14/'E14_matched_predictions.csv',index=False)
 pd.DataFrame([dict(weighting='equal_7_modules',raw_Brier=x.raw_Brier.mean(),platt_Brier=x.platt_Brier.mean(),prior_Brier=x.prior_Brier.mean(),delta_platt_minus_prior=(x.platt_Brier-x.prior_Brier).mean(),new_bootstrap_status='not_run_initial_batch')]).to_csv(E14/'E14_baseline_summary.csv',index=False)
 pd.DataFrame(CHECKS).to_csv(AUD/'baseline_reconciliation.csv',index=False)
 print('BASELINES PASS',len(CHECKS),flush=True)

def exports():
 rows=[];meta=[];predictdir=E10/'paired_predictions';predictdir.mkdir(exist_ok=True)
 featureframes={h:read(E3/f'day{h}_features.csv') for h in [28,56]}
 def add(env,task,family,domain,view,paths,score,params,trainframes):
  aa=[]
  for p in paths:
   q=norm(read(p),task,score)
   if family=='XGBoost' and env=='LOMO':q['p']=q.p.astype(np.float32).astype(float)
   aa.append(q)
  a,b=aa;z=pair(a,b);rid=f'E10A_{env}_{task}_{domain}_{family}_{view}'
  dest=predictdir/(rid+'.csv');z.to_csv(dest,index=False)
  for h,d,par,tr in zip([28,56],aa,params,trainframes):
   keys=tr['enrolment_key'] if 'enrolment_key' in tr else tr['key'] if 'key' in tr else tr.code_module+'|'+tr.code_presentation+'|'+tr.id_student.astype(str)
   ids=tr.id_student if 'id_student' in tr else tr.id
   assert not set(ids)&set(d.id_student)
   param=load(par);hp=param.get('params',param)
   meta.append(dict(experiment_id=rid,task=task,outcome_coding='Fail=1; Pass/Distinction=0, eventual withdrawals excluded' if task=='failure' else 'future Withdrawn=1; all other final outcomes=0',population_rule=view,source_rule=env,train_keys_hash=keyhash(keys),target_keys_hash=keyhash(z.record_id),feature_cutoff=h,family=family,hyperparam_hash=ohash(hp),seed=20261007,prediction_file=str(dest.relative_to(R)),historical_prediction=str(paths[(h==56)] .relative_to(R)),historical_parameters=str(par.relative_to(R)),domain=domain,N=len(z),positives=int(z.y.sum()),prevalence=z.y.mean(),train_N=len(tr),source_target_student_overlap=0))
  for viewname,x,y in [('dynamic_unpaired',a,b),(view,z.rename(columns={'pred_day28':'p'}),z.rename(columns={'pred_day56':'p'}))]:
   if view=='four_landmark_common' and viewname=='dynamic_unpaired':continue
   ma,mb=metrics(x),metrics(y)
   rows.append(dict(experiment_id=rid,environment=env,task=task,family=family,domain=domain,population=viewname,**{'day28_'+k:v for k,v in ma.items()},**{'day56_'+k:v for k,v in mb.items()},delta_nAP=mb['nAP']-ma['nAP'],delta_AP=mb['AP']-ma['AP'],pairing='no' if viewname=='dynamic_unpaired' else 'enrollment_key_and_label_verified',bootstrap_status='pending_2000_new_draws',train_params_policy='original_per_horizon; not fixed configuration'))
 for task in ['failure','withdrawal']:
  tr=[featureframes[h].query("split=='train'").dropna(subset=['y_'+task]) for h in [28,56]]
  for f in FAMILIES:
   pp=[A/f'experiment1_day28/models/{task}_{f}_parameters.json',E3/f'day56/models/{task}_{f}_parameters.json']
   for view,suffix in [('pairwise_common','test'),('four_landmark_common','stable_test')]:add('internal',task,f,'all',view,[E3/f'day{h}/{task}_{suffix}_predictions.csv' for h in [28,56]],f+'_probability',pp,tr)
   for m in ['AAA','BBB','CCC','DDD','EEE','FFF','GGG']:
    dirs=[root/f'folds/{m}/{task}' for root in [L28,L56]]
    add('LOMO',task,f,m,'pairwise_common',[d/f/'test_predictions.csv' for d in dirs],'probability',[d/f/'selection.json' for d in dirs],[read(d/'train_membership.csv') for d in dirs])
 for f in FAMILIES:
  dirs=[T/'Day28_failure',T/'E4T-B_day56_failure_secondary'];add('year2014','failure',f,'all','pairwise_common',[d/f/'test_predictions.csv' for d in dirs],'probability',[d/f/'selection.json' for d in dirs],[read(d/'train.csv') for d in dirs])
 pd.DataFrame(rows).to_csv(E10/'E10A_three_environment_family_matrix.csv',index=False);pd.DataFrame(meta).to_csv(E10/'E10_pairings_metadata.csv',index=False)
 # Source records and hyperparameters differ in layer A; only report, never assert fixed controls here.
 checks=[]
 for rid,z in pd.DataFrame(meta).groupby('experiment_id'):
  checks.append(dict(experiment_id=rid,paired_N=int(z.N.iloc[0]),target_keys_equal=z.target_keys_hash.nunique()==1,training_keys_equal=z.train_keys_hash.nunique()==1,hyperparameters_equal=z.hyperparam_hash.nunique()==1,source_target_student_overlap_max=int(z.source_target_student_overlap.max()),layer='A: historical same-family, not fixed configuration'))
 pd.DataFrame(checks).to_csv(E10/'E10_cohort_and_hp_audit.csv',index=False)
 print('E10 EXPORTS',len(checks),'pairs',len(rows),'point rows',flush=True)

def calendar_audit():
 raw=R/'archived_E1_E9/raw_data/OULAD DATASET.zip'
 with zipfile.ZipFile(raw) as z:
  courses=read(z.open('courses.csv'));info=read(z.open('studentInfo.csv'));assess=read(z.open('assessments.csv'))
 url='https://oro.open.ac.uk/53873/1/Untitled.pdf'
 def bounds(pres):
  year=int(pres[:4]);month={'B':2,'J':10}[pres[-1]];return datetime.date(year,month,1),datetime.date(year,month,calendar.monthrange(year,month)[1])
 rows=[]
 for q in courses.itertuples():
  lo,hi=bounds(q.code_presentation);length=int(q.module_presentation_length)
  rows.append(dict(code_module=q.code_module,code_presentation=q.code_presentation,module_presentation_length=length,start_date_exact='',start_earliest=lo,start_latest=hi,end_earliest=lo+datetime.timedelta(days=length),end_latest=hi+datetime.timedelta(days=length),final_label_available_by='',final_label_release_evidence='not present in released schema or reviewed descriptor',evidence_reference=url+' (pages 2 and 5-6); raw courses.csv',assumption_level='month established; exact day and label release unknown'))
 course=pd.DataFrame(rows);course.to_csv(E12/'E12_course_calendar_inventory.csv',index=False)
 sources=course[course.code_presentation.str.startswith('2013')];targets=course[course.code_presentation.str.startswith('2014')];comparisons=[]
 for s in sources.itertuples():
  for t in targets.itertuples():
   for h in [28,56]:
    for lag in [0,30,60]:
     cutofflo=t.start_earliest+datetime.timedelta(days=h);cutoffhi=t.start_latest+datetime.timedelta(days=h)
     plo=s.end_earliest+datetime.timedelta(days=lag);phi=s.end_latest+datetime.timedelta(days=lag)
     eligible=phi<cutofflo;status='proxy_eligible_all_start_days' if eligible else 'proxy_ineligible_all_start_days' if plo>=cutoffhi else 'proxy_ambiguous_start_day'
     comparisons.append(dict(source_module=s.code_module,source_presentation=s.code_presentation,target_module=t.code_module,target_presentation=t.code_presentation,target_landmark_day=h,source_start_evidence='month bounds only',source_start_earliest=s.start_earliest,source_start_latest=s.start_latest,source_end_earliest=s.end_earliest,source_end_latest=s.end_latest,source_final_label_available_by='',target_start_earliest=t.start_earliest,target_start_latest=t.start_latest,target_cutoff_earliest=cutofflo,target_cutoff_latest=cutoffhi,source_label_available_before_cutoff='UNKNOWN',proxy_release_lag_days=lag,proxy_label_earliest=plo,proxy_label_latest=phi,proxy_eligible_for_training=eligible,proxy_status=status,assumption_level='course end plus assumed lag; not observed release',evidence_reference=url))
 cal=pd.DataFrame(comparisons);cal.to_csv(E12/'E12_calendar_audit.csv',index=False);cal.to_csv(E12/'E12_asof_training_eligibility.csv',index=False)
 blacklist=set(info.loc[info.code_presentation.str.startswith('2014'),'id_student']);audit=[]
 for h,ex in [(28,'Day28_failure'),(56,'E4T-B_day56_failure_secondary')]:
  tr=read(T/ex/'train.csv');assert not set(tr.id)&blacklist
  for pres in ['2014B','2014J']:
   for lag in [0,30,60]:
    q=cal.query('target_presentation==@pres and target_landmark_day==@h and proxy_release_lag_days==@lag')
    allowed=set(q.loc[q.proxy_eligible_for_training,'source_module']+'_'+q.loc[q.proxy_eligible_for_training,'source_presentation'])
    sub=tr[tr.domain.isin(allowed)]
    audit.append(dict(horizon=h,target_presentation=pres,proxy_lag_days=lag,source_domains=';'.join(sorted(allowed)),source_domain_count=len(allowed),original_source_N=len(tr),proxy_source_N=len(sub),proxy_source_students=sub.id.nunique(),raw_2014_student_overlap=len(set(sub.id)&blacklist),verified_asof_status='UNVERIFIED_NO_LABEL_RELEASE_TIMESTAMPS',fit_status='not_run_initial_batch'))
 pd.DataFrame(audit).to_csv(E12/'E12_source_domain_count_and_overlaps.csv',index=False)
 pd.DataFrame([dict(analysis='original_retrospective',status='archived_E4T_reconciled',metrics_source='E4T_OULAD_temporal_generalization_v1.0'),dict(analysis='verified_calendar_asof',status='not_identifiable_without_release_evidence',metrics_source=''),dict(analysis='proxy_calendar_asof',status='eligibility_audited_training_not_run',metrics_source='')]).to_csv(E12/'E12_original_vs_verified_vs_proxy_metrics.csv',index=False)
 dump(E12/'evidence_and_policy.json',dict(source=url,reviewed_date='2026-10-09',evidence=['Descriptor page 2 establishes presentation start month from code.','Descriptor page 5 defines length as days from start to end.','Released studentInfo has final_result but no publication timestamp.','Assessment due/submission days do not establish final-result release.'],proxy_policy='Month bounds; length added to start; assumed lag 0/30/60 days; require latest proxy release strictly earlier than earliest target cutoff. Unknown factual releases remain UNKNOWN.',future_action='A proxy rerun must restrict all source-side development and calibration, not only final-fit records.'))
 print('E12 AUDIT',len(course),'presentations',len(cal),'scenario rows',flush=True)

def main():
 assert (AUD/'historical_SHA256.json').exists(),'Complete snapshot first'
 baseline();exports();calendar_audit()
 dump(AUD/'first_batch_execution.json',{'completed':['archive_copy_and_readonly_SHA256','E3_internal_reconciliation','E4T_year_reconciliation','E6C_and_source_prior_reconciliation','E12_calendar_audit','E10_three_family_prediction_exports'],'new_training_fits':0,'new_optuna_trials':0,'new_bootstrap_draws':0,'next_gate':'E10A paired bootstrap and E10B fixed training/parameters; new phase required in run log, not user reauthorization.'})
if __name__=='__main__':main()
