"""Clean raw-ZIP E4T failure reproduction. Run from a fresh venv with frozen requirements."""
import sys,os
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
from pathlib import Path
import argparse,json,zipfile,shutil,subprocess,hashlib,importlib.metadata,gc
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def main():
 a=argparse.ArgumentParser();a.add_argument('--raw',required=True);a.add_argument('--destination',required=True);a.add_argument('--reference',required=True);v=a.parse_args();raw=Path(v.raw);dest=Path(v.destination).resolve();ref=Path(v.reference).resolve()
 assert not dest.exists();assert sys.prefix!=sys.base_prefix;assert 'include-system-site-packages = false' in (Path(sys.prefix)/'pyvenv.cfg').read_text();assert sha(raw)=='f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4'
 versions={}
 for line in (HERE/'requirements-lock.txt').read_text().splitlines():
  if '==' in line:
   n,want=line.split('==');versions[n]=importlib.metadata.version(n);assert versions[n]==want,(n,versions[n],want)
 dest.mkdir(parents=True)
 for name in ['run.py','e1_reference.py']:shutil.copy2(HERE/'frozen_code'/name,dest/name)
 sys.path.insert(0,str(dest));import e1_reference as e1
 dump(dest/'clean_environment.json',dict(python=sys.version,prefix=sys.prefix,system_site_packages=False,versions=versions,raw_sha256=sha(raw),inputs='original ZIP and frozen code only; no frozen feature/model/split/trial inputs'))
 with zipfile.ZipFile(raw) as z:info=e1.read(z,'studentInfo.csv');reg=e1.read(z,'studentRegistration.csv');sites=e1.read(z,'vle.csv')
 held=set(info.loc[info.code_presentation.str.startswith('2014'),'id_student']);pd.DataFrame({'id_student':sorted(held)}).to_csv(dest/'raw_2014_student_blacklist.csv',index=False)
 chunks=[]
 with zipfile.ZipFile(raw) as z:
  for chunk in pd.read_csv(z.open('studentVle.csv'),na_values=['?'],chunksize=500000):chunks.append(chunk.loc[chunk.date.between(0,56)])
 logs=pd.concat(chunks,ignore_index=True);del chunks;cfg={}
 def members(r,role,z):z[['row_id','id','key','domain','module','y']].to_csv(r/(role+'.csv'),index=False)
 for h in [28,56]:
  cohort,allrecords=e1.active_cohort(info,reg,h);d,audit=e1.features_from_logs(cohort,logs,sites,h);allrecords[e1.KEY+['cohort_status']].to_csv(dest/f'day{h}_cohort_audit.csv',index=False);d.to_csv(dest/f'raw_day{h}_features.csv',index=False);dump(dest/f'day{h}_feature_audit.json',audit)
  d['id']=d.id_student;d['key']=d.code_module+'|'+d.code_presentation+'|'+d.id.astype(str);d['domain']=d.code_module+'_'+d.code_presentation;d['module']=d.code_module
  source=d[d.code_presentation.str.startswith('2013')&~d.id.isin(held)];target=d[d.code_presentation.str.startswith('2014')];exp='Day28_failure' if h==28 else 'E4T-B_day56_failure_secondary';r=dest/exp;r.mkdir();valid=d[d.y_failure.notna()].copy();valid['y']=valid.y_failure.astype(int);valid.to_csv(r/'data.csv',index=False);tr=valid[valid.row_id.isin(source.row_id)];te=valid[valid.row_id.isin(target.row_id)]
  for role,z in [('source',tr),('train',tr),('test',te)]:members(r,role,z)
  folds=sorted(source.domain.unique())
  for g in folds:
   ids=set(source.loc[source.domain.eq(g),'id']);members(r,'inner_'+g+'_train',tr[~tr.domain.eq(g)&~tr.id.isin(ids)]);members(r,'inner_'+g+'_validation',tr[tr.domain.eq(g)])
  cfg[exp]=dict(data=exp+'/data.csv',features=e1.FEATURES,categorical=[],task='failure',horizon=h,trials=20 if h==28 else 15,folds=folds,domain_disjoint=True,validation_union='source',matched_modules=sorted(set(source.module)&set(target.module)))
  print('RAW FEATURES AND SPLITS',h,len(d),len(tr),len(te),flush=True)
 del logs;gc.collect();dump(dest/'config.json',cfg);hashes={str(p.relative_to(dest)).replace('\\','/'):sha(p) for p in dest.rglob('*') if p.is_file()};dump(dest/'run_manifest.json',dict(hashes=hashes,seed=e1.SEED))
 for args in [['preaudit'],['train','--workers','3'],['score']]:subprocess.run([sys.executable,'-u',str(dest/'run.py'),*args],check=True)
 # Audit references are first read after the full pipeline has finished scoring.
 checks=[]
 for exp,c in cfg.items():
  h=c['horizon'];new=pd.read_csv(dest/f'raw_day{h}_features.csv',float_precision='round_trip');old=pd.read_csv(ref/f'frozen_day{h}_features.csv',float_precision='round_trip')
  for col in new:
   if pd.api.types.is_numeric_dtype(new[col]):assert np.allclose(new[col],old[col],atol=1e-12,rtol=1e-12,equal_nan=True),(h,col)
   else:assert new[col].equals(old[col]),(h,col)
  for p in (dest/exp).glob('*.csv'):
   if p.name!='data.csv':pd.testing.assert_frame_equal(pd.read_csv(p),pd.read_csv(ref/exp/p.name))
  for name in ['LR','XGBoost','CatBoost']:
   x=json.loads((dest/exp/name/'selection.json').read_text());y=json.loads((ref/exp/name/'selection.json').read_text());assert x['params']==y['params'] and x['best_trial']==y['best_trial'];assert abs(x['inner_mean_AP']-y['inner_mean_AP'])<1e-10
   aa=pd.read_csv(dest/exp/name/'trials.csv');bb=pd.read_csv(ref/exp/name/'trials.csv');assert np.allclose(aa.value,bb.value,atol=1e-10,rtol=1e-10)
   p=pd.read_csv(dest/exp/name/'test_predictions.csv',float_precision='round_trip');q=pd.read_csv(ref/exp/name/'test_predictions.csv',float_precision='round_trip');assert p.key.equals(q.key);delta=float(np.max(abs(p.probability-q.probability)));assert delta<1e-9;checks.append(dict(experiment=exp,model=name,trials=len(aa),same_best_parameters=True,max_prediction_difference=delta))
 selected=pd.read_csv(dest/'source_selected_models.csv');old=pd.read_csv(ref/'source_selected_models.csv');assert selected.set_index('experiment').selected_model.equals(old[old.experiment.isin(cfg)].set_index('experiment').selected_model)
 ps=[pd.read_csv(dest/ex/'LR/test_predictions.csv',float_precision='round_trip') for ex in cfg];z=ps[0].merge(ps[1],on='key',suffixes=('_28','_56'),validate='one_to_one');from sklearn.metrics import average_precision_score
 pi=z.y_28.mean();gain=(average_precision_score(z.y_56,z.probability_56)-average_precision_score(z.y_28,z.probability_28))/(1-pi);old=pd.read_csv(ref/'paired_horizon_comparison.csv');want=old[old.scope.eq('all')&old.comparison.eq('LR')].delta_nAP.iloc[0];assert abs(gain-want)<1e-10
 dump(dest/'raw_reproduction_audit.json',dict(status='passed',new_clean_environment=True,raw_features_and_splits_reconstructed=True,frozen_data_used_for_training=False,trials=105,inner_fits=945,final_fits=6,checks=checks,selected_models=selected.to_dict('records'),common_N=len(z),common_gain_nAP=gain,frozen_gain_nAP=want));print('RAW ZIP CLEAN REPRODUCTION PASSED',gain,flush=True)
if __name__=='__main__':main()
