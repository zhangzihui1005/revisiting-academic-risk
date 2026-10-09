"""Reload saved estimators and independently check predictions and source isolation."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
R=Path(__file__).resolve().parents[1];A=R/'archived_E1_E9/outputs';AUD=R/'revision_outputs/audit';NEW=R/'experiments_new'
def read(p):return pd.read_csv(p,float_precision='round_trip')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 rows=[];plan=load(AUD/'fixed_controls_plan.json')
 for j in plan['jobs']:
  tr=read(j['train']);te=read(j['test']);assert not set(tr.id)&set(te.id)
  if j['experiment']=='E13':
   base=A/('E5A_UCI_information_horizons_v1.0' if j['environment']=='internal' else 'E5B_UCI_LOCO_v1.0');old=read(base/str(j['horizon'])/'train.csv');assert np.array_equal(tr.row_id,old.row_id)
   assert not set(['GDP','Unemployment rate','Inflation rate'])&set(j['features'])
  m=joblib.load(Path(j['dest'])/'model.joblib');p=m.predict_proba(te[j['features']])[:,1].astype(float);stored=read(Path(j['dest'])/'predictions.csv');assert np.array_equal(te.key,stored.record_id);err=float(np.max(np.abs(p-stored.p)));assert err<1e-12
  rows.append(dict(job=j['id'],N=len(te),max_probability_error=err,source_target_overlap=0,status='PASS'))
 for j in load(NEW/'E12_calendar_audit/proxy_prerun_plan.json')['jobs']:
  tr=read(j['source']);assert tr.domain.str.endswith('2013B').all();dest=Path(j['dest']);pred=read(NEW/'E12_calendar_audit/proxy_predictions'/f"{j['experiment']}_{j['family']}_2014B.csv")
  d=read(A/'E4T_OULAD_temporal_generalization_v1.0'/j['experiment']/'data.csv').set_index('key');te=d.loc[pred.key];assert not set(tr.id)&set(te.id)
  features=load(AUD/'fixed_controls_plan.json')['jobs'][0]['features'];m=joblib.load(dest/'model.joblib');p=m.predict_proba(te[features])[:,1];err=float(np.max(np.abs(p-pred.raw)));assert err<1e-12
  rows.append(dict(job='E12_'+j['experiment']+'_'+j['family'],N=len(te),max_probability_error=err,source_target_overlap=0,status='PASS'))
 pd.DataFrame(rows).to_csv(AUD/'saved_model_prediction_replay.csv',index=False)
 print('REPLAY PASS',len(rows),'saved models; full target predictions checked')
if __name__=='__main__':main()
