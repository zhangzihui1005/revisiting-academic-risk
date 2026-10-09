"""Source-only ensemble gate. Target stage runs only after the gate decision is locked."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,time,shutil
import numpy as np,pandas as pd,joblib
from sklearn.linear_model import LogisticRegression
import e6_common as c
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
METHODS=c.NAMES+['S1_selected','S2_uniform','S3_two_tree','S4_stacking']
def meta():return LogisticRegression(C=1.,solver='lbfgs',max_iter=4000,random_state=c.SEED)
def source():
 assert not (ROOT/'source_gate_decision.json').exists();specs=c.specs(PARENT);rows=[];aud=[];hashes={};locks={}
 for s in specs:
  base=Path(s['base']);dest=ROOT/s['key'];dest.mkdir(parents=True,exist_ok=True)
  for n in c.NAMES:
   for fn in ['best_inner_predictions.csv.gz','selection.json']:
    p=base/n/fn;hashes[str(p.relative_to(PARENT)).replace('\\','/')]=c.sha(p)
  z=c.aligned(s,'source');z['S1_selected']=z[s['selected']];z['S2_uniform']=z[c.NAMES].mean(axis=1);z['S3_two_tree']=z[['XGBoost','CatBoost']].mean(axis=1);z['S4_stacking']=np.nan
  for g in sorted(z.domain.unique()):
   val=z.domain.eq(g);held=set(z.loc[val,'id']);train=~val&~z.id.isin(held);m=meta();m.fit(z.loc[train,c.NAMES],z.loc[train,'y']);z.loc[val,'S4_stacking']=m.predict_proba(z.loc[val,c.NAMES])[:,1]
   joblib.dump(m,dest/('meta_without_'+g+'.joblib'))
   aud.append(dict(experiment=s['key'],validation_domain=g,train_N=int(train.sum()),validation_N=int(val.sum()),student_overlap=len(set(z.loc[train,'id'])&held),base_predictions='frozen source OOF; meta cross-fit is not full base-model nesting'))
  assert z.S4_stacking.notna().all();m=meta();m.fit(z[c.NAMES],z.y);joblib.dump(m,dest/'meta_final.joblib');z.to_csv(dest/'source_predictions.csv.gz',index=False)
  c.dump(dest/'stacking_parameters.json',dict(features=c.NAMES,intercept=m.intercept_.tolist(),coef=m.coef_.tolist(),C=1.,class_weight=None,standardization=False))
  rows+=c.domain_rows(z,METHODS,dict(setting=s['setting'],environment=s['environment'],outer=s['outer'],experiment=s['key']))
  for p in dest.iterdir():locks[str(p.relative_to(ROOT)).replace('\\','/')]=c.sha(p)
  print('SOURCE LOCKED',s['key'],flush=True)
 pd.DataFrame(aud).to_csv(ROOT/'meta_crossfit_audit.csv',index=False);pd.DataFrame(rows).to_csv(ROOT/'source_domain_metrics.csv',index=False);summary=c.summarize(rows);summary.to_csv(ROOT/'source_summary.csv',index=False)
 gate=[]
 for setting,z in summary.groupby('setting'):
  singles=z[z.method.isin(c.NAMES)];bestmean=singles.Mean_nAP.max();bestworst=singles.Min_nAP.max()
  for method in ['S2_uniform','S4_stacking']:
   row=z[z.method.eq(method)].iloc[0];dm=row.Mean_nAP-bestmean;dw=row.Min_nAP-bestworst;passes=(dm>1e-12 and dw>=-.005-1e-12) or (dw>1e-12 and dm>=-.005-1e-12)
   gate.append(dict(setting=setting,environment=setting[0],method=method,best_single_mean_nAP=bestmean,best_single_worst_nAP=bestworst,mean_nAP=row.Mean_nAP,worst_nAP=row.Min_nAP,delta_mean=dm,delta_worst=dw,passes=bool(passes)))
 gd=pd.DataFrame(gate);gd.to_csv(ROOT/'gate_by_setting.csv',index=False);envs={n:sorted(gd.loc[gd.passes&gd.method.eq(n),'environment'].unique()) for n in ['S2_uniform','S4_stacking']};decision=dict(MOPSO_GO=any(len(v)>=2 for v in envs.values()),qualifying_environments=envs,required_distinct_environments=2,target_used_for_decision=False,noninferiority_tolerance_nAP=.005,strict_gain_tolerance=1e-12,benchmark='separate max Mean_nAP and max Worst_nAP over fixed LR/XGB/Cat families; conservative source single-model envelope',aggregation='equally weighted outer x source-domain cells within each setting; Worst=min cell',same_ensemble_required=True,stacking_source_limit='domain-crossfit meta learner conditions on reused/tuned base OOF; not a fully nested unbiased source estimate',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
 c.dump(ROOT/'source_gate_decision.json',decision);c.dump(ROOT/'source_input_hashes.json',hashes);c.dump(ROOT/'source_lock.json',dict(specs=specs,files=locks,gate_sha256=c.sha(ROOT/'source_gate_decision.json'),protocol_sha256=c.sha(ROOT/'PROTOCOL.txt'),code_sha256=c.sha(Path(__file__)),common_sha256=c.sha(ROOT/'e6_common.py')));print(decision,flush=True)
def target():
 lock=c.load(ROOT/'source_lock.json');assert c.sha(ROOT/'source_gate_decision.json')==lock['gate_sha256'];assert c.sha(ROOT/'PROTOCOL.txt')==lock['protocol_sha256'];assert c.sha(Path(__file__))==lock['code_sha256'];rows=[];pooled=[]
 for s in lock['specs']:
  dest=ROOT/s['key'];z=c.aligned(s,'target');o=pd.read_csv(dest/'source_predictions.csv.gz');assert not set(z.id)&set(o.id)
  m=joblib.load(dest/'meta_final.joblib');z['S1_selected']=z[s['selected']];z['S2_uniform']=z[c.NAMES].mean(axis=1);z['S3_two_tree']=z[['XGBoost','CatBoost']].mean(axis=1);z['S4_stacking']=m.predict_proba(z[c.NAMES])[:,1];z.to_csv(dest/'target_predictions.csv',index=False)
  # B uses target course modules to make worst-domain results interpretable.
  if s['environment']=='B':z['domain']=z.module
  rows+=c.domain_rows(z,METHODS,dict(setting=s['setting'],environment=s['environment'],outer=s['outer'],experiment=s['key']))
  for n in METHODS:pooled.append(dict(setting=s['setting'],experiment=s['key'],method=n,**c.metrics(z.y,z[n])))
 pd.DataFrame(rows).to_csv(ROOT/'target_domain_metrics.csv',index=False);c.summarize(rows).to_csv(ROOT/'target_summary.csv',index=False);pd.DataFrame(pooled).to_csv(ROOT/'target_outer_pooled_metrics.csv',index=False)
 for p,h in c.load(ROOT/'source_input_hashes.json').items():assert c.sha(PARENT/p)==h,p
 for p,h in lock['files'].items():assert c.sha(ROOT/p)==h,p
 c.dump(ROOT/'execution_checks.json',dict(status='passed',outer_pipelines=len(lock['specs']),base_refits=0,meta_student_disjoint=True,target_not_used_for_gate=True,source_files_unchanged=True))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['source','target']);v=a.parse_args();source() if v.stage=='source' else target()
