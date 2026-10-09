"""Reporting/validation only. All inference reads already saved replicates."""
import sys,os
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,datetime,zipfile
import pandas as pd,numpy as np
from sklearn.metrics import average_precision_score
R=Path(__file__).resolve().parents[1];NEW=R/'experiments_new';AUD=R/'revision_outputs/audit';TABLE=R/'revision_outputs/tables';TABLE.mkdir(exist_ok=True)
def read(p):return pd.read_csv(p,float_precision='round_trip')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def normalize_and_validate_bootstrap():
 rows=[];aud=[];ren={'AP_day28':'AP_earlier','AP_day56':'AP_later','nAP_day28':'nAP_earlier','nAP_day56':'nAP_later'}
 paths=list((NEW/'E10_temporal_control/bootstrap_E10A').glob('*/summary.json'))
 for name in ['E10_temporal_control','E11_full_risk_set','E12_calendar_audit','E13_UCI_availability']:paths+=list((NEW/name/'bootstrap_new').glob('*/summary.json'))
 assert len(paths)==79
 for p in sorted(paths):
  s=load(p);inp=R/s['input'];d=read(inp);rep=p.parent/'replicates.csv';z=read(rep)
  if s['input_sha256']!=sha(inp):
   # One intentional metadata-only correction: UCI horizon names were never OULAD days.
   assert s['analysis'].startswith('E13_');original_bytes=inp.read_bytes().replace(b'pred_earlier',b'pred_day28',1).replace(b'pred_later',b'pred_day56',1);assert hashlib.sha256(original_bytes).hexdigest()==s['input_sha256']
   s['input_before_header_only_rename_sha256']=s['input_sha256'];s['input_sha256']=sha(inp)
   rng=load(p.parent/'rng_state.json');rng['input_header_only_rename']='UCI pred_day28/pred_day56 aliases replaced by pred_earlier/pred_later; numeric CSV body unchanged';dump(p.parent/'rng_state.json',rng)
  for old,new in ren.items():
   if old in s:s[new]=s.pop(old)
  if any(c in z for c in ren):z=z.rename(columns=ren);z.to_csv(rep,index=False)
  s['earlier_horizon']='H0_strict' if 'H0_strict_to_H1' in s['analysis'] else 'H1' if s['analysis'].startswith('E13_') else 'Day28';s['later_horizon']='H1' if 'H0_strict_to_H1' in s['analysis'] else 'H2' if s['analysis'].startswith('E13_') else 'Day56'
  s['bootstrap_unit']='released UCI row identity' if s['analysis'].startswith('E13_') else 'global OULAD student ID';dump(p,s)
  a='pred_earlier' if 'pred_earlier' in d else 'pred_day28';b='pred_later' if 'pred_later' in d else 'pred_day56'
  assert len(d)==s['N'] and not d.record_id.duplicated().any();pi=d.y.mean();delta=(average_precision_score(d.y,d[b])-average_precision_score(d.y,d[a]))/(1-pi);assert abs(delta-s['delta_nAP'])<1e-12
  valid=z[z.valid];lo,hi=valid.delta_nAP.quantile([.025,.975]);assert len(z)==2000 and len(valid)==s['draws_valid'];assert abs(lo-s['CI95_low'])<1e-12 and abs(hi-s['CI95_high'])<1e-12
  # Independently replay one saved draw from stored RNG state and cluster order.
  states=load(p.parent/'rng_state.json');rng=np.random.default_rng();rng.bit_generator.state=states['initial'];groups=read(p.parent/'cluster_order.csv').id_student.to_numpy();codes=pd.Index(groups).get_indexer(d.id_student);assert (codes>=0).all();w=np.bincount(rng.integers(0,len(groups),size=len(groups)),minlength=len(groups))[codes];pp=np.average(d.y,weights=w);da=(average_precision_score(d.y,d[b],sample_weight=w)-average_precision_score(d.y,d[a],sample_weight=w))/(1-pp);assert abs(da-z.delta_nAP.iloc[0])<1e-12
  rows.append(s);aud.append(dict(analysis=s['analysis'],N=len(d),replicates=len(z),valid=len(valid),observed_error=abs(delta-s['delta_nAP']),rng_replay_error=abs(da-z.delta_nAP.iloc[0]),status='PASS'))
 pd.DataFrame(aud).to_csv(AUD/'new_bootstrap_independent_audit.csv',index=False);allrows=pd.DataFrame(rows);allrows.to_csv(TABLE/'S26_E10_E14_paired_horizon_comparisons.csv',index=False)
 allrows[allrows.analysis.str.startswith('E10A')].to_csv(R/'revision_outputs/paired_bootstrap_initial_summary.csv',index=False);allrows[~allrows.analysis.str.startswith('E10A')].to_csv(R/'revision_outputs/paired_bootstrap_fixed_summary.csv',index=False)
 return allrows
def e12_comparison():
 root=NEW/'E12_calendar_audit';new=read(root/'E12_proxy_results.csv');original=[];A=R/'archived_E1_E9/outputs'
 for ex in ['Day28_failure','Day28_withdrawal','E4T-B_day56_failure_secondary']:
  d=read(A/'E6C_probability_transportability_v1.0'/('B_'+ex)/'target_predictions.csv')
  for pres in ['2014B','2014J','2014B_and_2014J']:
   z=d if pres=='2014B_and_2014J' else d[d.domain.str.endswith(pres)]
   for method in ['raw','calibrated','source_prior']:
    pi=z.y.mean();ap=average_precision_score(z.y,z[method]);original.append(dict(experiment=ex,target_presentation=pres,family='archived_source_selected',source_selected=True,method=method,analysis='original_retrospective',N=len(z),positive=int(z.y.sum()),prevalence=pi,AP=ap,nAP=(ap-pi)/(1-pi),Brier=np.mean((z.y-z[method])**2),calendar_verified=False))
 new['calendar_verified']=False;d=pd.concat([pd.DataFrame(original),new],ignore_index=True);d.to_csv(root/'E12_original_vs_verified_vs_proxy_metrics.csv',index=False);d.to_csv(TABLE/'S29_E12_original_and_proxy_results.csv',index=False)
def main():
 stats=normalize_and_validate_bootstrap();e12_comparison()
 E14=NEW/'E14_calibration_baseline';d=read(E14/'E14_matched_predictions.csv');rep=read(E14/'bootstrap/E14_equal_module_global_student_bootstrap.csv');s=read(E14/'E14_paired_comparisons.csv');assert len(rep)==5000
 for row in s.itertuples():
  lo,hi=rep[row.contrast].quantile([.025,.975]);assert abs(lo-row.CI95_low)<1e-12 and abs(hi-row.CI95_high)<1e-12
 rng=np.random.default_rng();rng.bit_generator.state=load(E14/'bootstrap/rng_state.json')['initial'];groups=read(E14/'bootstrap/cluster_order.csv').id_student.to_numpy();counts=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups));w=counts[pd.Index(groups).get_indexer(d.id_student)];d['weight']=w
 delta=d.groupby('code_module').apply(lambda q:np.average((q.y-q.p_platt)**2-(q.y-q.p_prior)**2,weights=q.weight)).mean();assert abs(delta-rep.equal7_delta_platt_minus_prior_Brier.iloc[0])<1e-12
 s.to_csv(TABLE/'S31_E14_calibration_vs_constant.csv',index=False)
 read(NEW/'E10_temporal_control/E10B_fixed_configuration_metrics.csv').to_csv(TABLE/'S27_E10B_fixed_configuration_metrics.csv',index=False)
 read(NEW/'E11_full_risk_set/E11_fullrisk_internal_dynamic.csv').to_csv(TABLE/'S28_E11_fullrisk_dynamic.csv',index=False)
 read(NEW/'E13_UCI_availability/E13_fixed_configuration_metrics.csv').to_csv(TABLE/'S30_E13_minus_macro_metrics.csv',index=False)
 # Keep optional analyses explicit rather than mark every manifest row as completed.
 manifest=read(R/'protocol_received/run_manifest.csv');manifest['status']='completed'
 manifest.loc[manifest.run_id.isin(['E11-YEAR','E14-UCI']),'status']='optional_not_run'
 manifest.loc[manifest.run_id.eq('E12-RETRAIN'),'status']='proxy_only_completed_verified_asof_unavailable'
 manifest.to_csv(R/'run_manifest_current.csv',index=False)
 e11=load(NEW/'E11_full_risk_set/execution_summary.json');e11['bootstrap']='complete_7_comparisons_2000_draws_each';e11['status']='complete_internal_primary_and_controls';dump(NEW/'E11_full_risk_set/execution_summary.json',e11)
 original=load(AUD/'historical_SHA256.json');changed=[q['original'] for q in original if sha(q['original'])!=q['sha256']];assert not changed
 fixed=read(NEW/'E10_temporal_control/E10B_fixed_train_fixed_params_audit.csv');assert fixed.status.eq('PASS').all()
 public={'status':'PASS','historical_files_checked':len(original),'historical_files_changed':changed,'new_horizon_comparisons':len(stats),'retained_new_horizon_draws':int(stats.draws_attempted.sum()),'new_calibration_global_cluster_draws':5000,'superseded_UCI_horizon_draws':12000,'new_Optuna_trials':195,'retained_workflow_model_fits':605,'superseded_UCI_model_fits':26,'actual_new_model_fits_including_superseded':631,'new_calibrator_fits':9,'E10B_controls':len(fixed),'E13_internal_fits':9,'E13_LOCO_courses':17,'calendar_release_evidence':'unknown; E12 proxy only','optional_not_run':['E11 year extension','E14 UCI calibration extension','E13 retuned and Course-removal variants'],'method':'No historical bootstrap reused as new inference. New intervals conditional on fixed models and records; descriptive, not familywise adjusted.'}
 dump(AUD/'E10_E14_final_validation.json',public)
 print(json.dumps(public,ensure_ascii=False))
if __name__=='__main__':main()
