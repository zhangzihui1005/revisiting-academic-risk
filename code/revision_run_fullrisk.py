"""E11 internal full active-risk-set evaluation, four prespecified arms."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='6'
from pathlib import Path
import json,hashlib,datetime,time
import numpy as np,pandas as pd,joblib,optuna
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];A=R/'archived_E1_E9/outputs';OUT=R/'experiments_new/E11_full_risk_set';E1=A/'experiment1_day28';E2=A/'E2_day28_formulation_v1.0';E3=A/'E3_oulad_multihorizon_v1.0'
sys.path.insert(0,str(E1));import run_experiment as e1
FEATURES=e1.FEATURES;SEED=e1.SEED
def read(p):return pd.read_csv(p,float_precision='round_trip')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def hashkeys(k):return hashlib.sha256(('\n'.join(sorted(k))+'\n').encode()).hexdigest()
def frame(h):
 d=read(E3/f'day{h}_features.csv');d['record_id']=d.code_module+'|'+d.code_presentation+'|'+d.id_student.astype(str);d['y_three']=d.final_result.map({'Pass':0,'Distinction':0,'Fail':1,'Withdrawn':2});assert d.y_three.notna().all();return d.sort_values('record_id').reset_index(drop=True)
def bmetrics(y,p,keys):
 pi=np.mean(y);ap=average_precision_score(y,p);idx=np.lexsort((np.asarray(keys,str),-np.asarray(p)));k=int(np.ceil(.1*len(y)));hits=np.asarray(y)[idx[:k]].sum();return dict(N=len(y),positive=int(np.sum(y)),prevalence=pi,AP=ap,nAP=(ap-pi)/(1-pi),AUROC=roc_auc_score(y,p),Brier=np.mean((y-p)**2),Precision_at_10=hits/k,Recall_at_10=hits/np.sum(y),Lift_at_10=hits/k/pi)
def main():
 start=time.time();OUT.mkdir(exist_ok=True);pred=OUT/'predictions';pred.mkdir(exist_ok=True);models=OUT/'models';models.mkdir(exist_ok=True)
 plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary':'Flat CatBoost, full active-risk-set Fail, paired common test 28-to-56','arms':['flat_CatBoost','dual_CatBoost_LightGBM','direct_Fail_CatBoost','dual_CatBoost_CatBoost'],'day28':'reuse archived flat/dual models; score all active records','day56':'same original family configs refit on dynamic source training rows','direct_fail_selection':'30 fresh source-train/source-validation Day28 CatBoost trials, frozen original search space and seed; AP objective; no target scoring until lock; selected params reused at Day56','new_experiment_not_historical_preregistration':True,'year_extension':'not run; internal primary with proxy-year analysis separately in E12','threads':6}
 if not (OUT/'prerun_plan.json').exists():dump(OUT/'prerun_plan.json',plan)
 d28=frame(28);tr=d28[d28.split.eq('train')];va=d28[d28.split.eq('validation')]
 assert not set(tr.id_student)&set(va.id_student)
 lock=OUT/'direct_fail_selection.json'
 if not lock.exists():
  optuna.logging.set_verbosity(optuna.logging.WARNING);study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED),storage='sqlite:///'+str(OUT/'direct_fail_study.sqlite3').replace('\\','/'),study_name='E11_direct_fail_source_validation',load_if_exists=True)
  def objective(trial):
   p=e1.parameters(trial,'CatBoost');m=e1.model('CatBoost',p)
   with threadpool_limits(limits=6):m.fit(tr[FEATURES],tr.y_three.eq(1).astype(int));s=m.predict_proba(va[FEATURES])[:,1]
   return average_precision_score(va.y_three.eq(1),s)
  study.optimize(objective,n_trials=max(0,30-len(study.trials)));study.trials_dataframe().to_csv(OUT/'direct_fail_source_trials.csv',index=False);dump(lock,dict(params=study.best_params,source_validation_AP=study.best_value,trial=study.best_trial.number,seed=SEED,train_keys_hash=hashkeys(tr.record_id),validation_keys_hash=hashkeys(va.record_id),target_seen_for_selection=False,trials=len(study.trials)))
  print('E11 SOURCE SELECTION LOCKED',flush=True)
 direct_params=load(lock)['params'];allmetrics=[];multirows=[];conf=[];aud=[];frames={};dedup=[]
 for h in [28,56]:
  d=frame(h);train=d[d.split.eq('train')];test=d[d.split.eq('test')];assert not set(train.id_student)&set(test.id_student)
  if h==28:assert len(test)==5485 and test.y_three.eq(2).sum()==1003 and test.y_three.eq(1).sum()==1406
  y=test.y_three.to_numpy();x=test[FEATURES];arms={};binary_models={}
  for branch,family in [('withdrawal','CatBoost'),('failure','LightGBM'),('failure','CatBoost')]:
   name=branch+'_'+family
   if h==28:m=joblib.load(E1/'models'/(name+'.joblib'))['model']
   else:
    params=load(E1/'models'/(name+'_parameters.json'));m=e1.model(family,params);eligible=train if branch=='withdrawal' else train[train.y_three.ne(2)];target=eligible.y_three.eq(2 if branch=='withdrawal' else 1).astype(int)
    with threadpool_limits(limits=6):m.fit(eligible[FEATURES],target)
    joblib.dump(m,models/f'D56_{name}.joblib',compress=3)
   # All target records are scored by both conditional-failure branches.
   binary_models[name]=m.predict_proba(x)[:,1].astype(float)
  if h==28:flat=joblib.load(E2/'models/flat_CatBoost.joblib')['model']
  else:
   flat=e1.model('CatBoost',load(E2/'models/flat_CatBoost_parameters.json'));flat.set_params(loss_function='MultiClass')
   with threadpool_limits(limits=6):flat.fit(train[FEATURES],train.y_three)
   joblib.dump(flat,models/'D56_flat_CatBoost.joblib',compress=3)
  arms['flat_CatBoost']=flat.predict_proba(x)[:,[list(flat.classes_).index(i) for i in range(3)]]
  w=binary_models['withdrawal_CatBoost']
  for name,f in [('dual_CatBoost_LightGBM',binary_models['failure_LightGBM']),('dual_CatBoost_CatBoost',binary_models['failure_CatBoost'])]:arms[name]=np.column_stack([(1-w)*(1-f),(1-w)*f,w])
  direct=e1.model('CatBoost',direct_params)
  with threadpool_limits(limits=6):direct.fit(train[FEATURES],train.y_three.eq(1).astype(int))
  joblib.dump(direct,models/f'D{h}_direct_Fail_CatBoost.joblib',compress=3);dp=direct.predict_proba(x)[:,1]
  for name,probs in list(arms.items())+[('direct_Fail_CatBoost',dp)]:
   z=test[['record_id','id_student','code_module','code_presentation','y_three']].copy();z['p_Fail']=probs[:,1] if probs.ndim==2 else probs
   if probs.ndim==2:
    assert np.allclose(probs.sum(axis=1),1,atol=1e-12) and np.isfinite(probs).all() and (probs>=0).all();z['p_Success']=probs[:,0];z['p_Withdrawal']=probs[:,2]
    yh=probs.argmax(axis=1);multirows.append(dict(horizon=h,arm=name,N=len(test),MulticlassBrier=np.mean(np.sum((probs-np.eye(3)[y])**2,axis=1)),MulticlassLogLoss=log_loss(y,probs,labels=[0,1,2]),MacroF1=f1_score(y,yh,average='macro'),Brier_convention='sum over three classes'))
    cm=confusion_matrix(y,yh,labels=[0,1,2])
    for i in range(3):
     for j in range(3):conf.append(dict(horizon=h,arm=name,true_class=['Success','Fail','Withdrawal'][i],predicted_class=['Success','Fail','Withdrawal'][j],N=int(cm[i,j])))
   z.to_csv(pred/f'E11_D{h}_{name}.csv',index=False);frames[(h,name)]=z
   for task,col in [('Fail','p_Fail')]+([('Withdrawal','p_Withdrawal')] if probs.ndim==2 else []):
    target=z.y_three.eq(1 if task=='Fail' else 2).astype(int).to_numpy();allmetrics.append(dict(horizon=h,arm=name,task=task,population='dynamic_full_active_risk_set',**bmetrics(target,z[col].to_numpy(),z.record_id)))
    # Capacity sensitivity: rank unique students by highest enrollment risk; positive if any enrollment is positive.
    uu=pd.DataFrame({'id_student':z.id_student,'y':target,'p':z[col]}).groupby('id_student',sort=True).agg(y=('y','max'),p=('p','max')).reset_index();dedup.append(dict(horizon=h,arm=name,task=task,aggregation='student max enrollment risk; event in any active enrollment',**bmetrics(uu.y.to_numpy(),uu.p.to_numpy(),uu.id_student.astype(str))))
   aud.append(dict(horizon=h,arm=name,test_N=len(test),test_keys_hash=hashkeys(z.record_id),train_N=len(train),train_keys_hash=hashkeys(train.record_id),future_outcome_filters_on_test=0,student_overlap=0,seed=SEED))
  print('E11 DAY',h,'COMPLETE',flush=True)
 pairs=OUT/'new_paired_predictions';pairs.mkdir(exist_ok=True);pairmetrics=[]
 for name in ['flat_CatBoost','dual_CatBoost_LightGBM','direct_Fail_CatBoost','dual_CatBoost_CatBoost']:
  a,b=frames[(28,name)],frames[(56,name)];z=a.merge(b,on=['record_id','id_student'],suffixes=('_28','_56'),validate='one_to_one');assert np.array_equal(z.y_three_28,z.y_three_56)
  for task,col in [('Fail','p_Fail')]+([('Withdrawal','p_Withdrawal')] if name!='direct_Fail_CatBoost' else []):
   yy=z.y_three_28.eq(1 if task=='Fail' else 2).astype(int);o=pd.DataFrame({'record_id':z.record_id,'id_student':z.id_student,'y':yy,'pred_day28':z[col+'_28'],'pred_day56':z[col+'_56']}).sort_values('record_id');o.to_csv(pairs/f'E11_{name}_{task}.csv',index=False)
   for h in [28,56]:pairmetrics.append(dict(arm=name,task=task,horizon=h,population='retrospective_pairwise_common_full_risk_set',**bmetrics(yy.to_numpy(),z[col+f'_{h}'].to_numpy(),z.record_id)))
 pd.DataFrame(allmetrics).to_csv(OUT/'E11_fullrisk_internal_dynamic.csv',index=False);pd.DataFrame(pairmetrics).to_csv(OUT/'E11_fullrisk_pairwise_common.csv',index=False);pd.DataFrame(multirows).to_csv(OUT/'E11_multiclass_metrics.csv',index=False);pd.DataFrame(conf).to_csv(OUT/'E11_confusion_matrices.csv',index=False);pd.DataFrame(aud).to_csv(OUT/'E11_fullrisk_audit.csv',index=False);pd.DataFrame(dedup).to_csv(OUT/'E11_student_deduplicated_capacity.csv',index=False)
 dump(OUT/'execution_summary.json',dict(status='fits_and_predictions_complete',new_optuna_trials=30,new_source_validation_fits=30,new_final_fits=6,archived_model_reuses=4,seconds=time.time()-start,bootstrap='pending',year_extension='not_run_optional'))
if __name__=='__main__':main()
