"""E4B uses the immutable E4A training worker with explicit Day56 configuration."""
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
from pathlib import Path
import argparse,json,shutil,zipfile,gc,itertools,concurrent.futures,subprocess,time
import numpy as np,pandas as pd,joblib
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
import e1_reference as e1
import e4a_engine as engine
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent;A=PARENT/'E4A_day28_student_disjoint_LOMO_v1.0'
MOD=[s*3 for s in 'ABCDEFG'];TASKS=['withdrawal','failure'];NAMES=['LR','XGBoost','CatBoost']
def dump(p,o):e1.dump(p,o)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def key(d):return d.code_module+'|'+d.code_presentation+'|'+d.id_student.astype(str)
def frame():
    d=pd.read_csv(ROOT/'day56_features.csv');d['enrolment_key']=key(d);return d
def integrity():
    m=load(ROOT/'run_manifest.json')
    for name,h in m['fixed_hashes'].items():assert e1.sha(ROOT/name)==h,name
    for name,h in m['split_hashes'].items():assert e1.sha(ROOT/name)==h,name
def frozen_check():
    n=0
    for group in load(ROOT/'frozen_sources.json').values():
        for rel,h in group['files'].items():assert e1.sha(Path(group['directory'])/rel)==h,(group['directory'],rel);n+=1
    return n
engine.ROOT=ROOT;engine.NAMES=NAMES;engine.frame=frame;engine.integrity=integrity;engine.frozen_check=frozen_check
def prepare(raw):
    assert not (ROOT/'run_manifest.json').exists()
    old=load(A/'frozen_sources.json');old['E4A']={'directory':str(A),'files':{str(p.relative_to(A)).replace('\\','/'):e1.sha(p) for p in sorted(A.rglob('*')) if p.is_file()}}
    dump(ROOT/'frozen_sources.json',old);frozen_check()
    assert e1.sha(raw)=='f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4'
    assert e1.sha(ROOT/'e4a_engine.py')==e1.sha(A/'run_e4a.py')
    with zipfile.ZipFile(raw) as z:info=e1.read(z,'studentInfo.csv');reg=e1.read(z,'studentRegistration.csv');sites=e1.read(z,'vle.csv')
    cohort,allrows=e1.active_cohort(info,reg,cutoff=56)
    allrows[e1.KEY+['date_registration','date_unregistration','final_result','cohort_status']].to_csv(ROOT/'day56_cohort_audit.csv',index=False)
    chunks=[]
    with zipfile.ZipFile(raw) as z:
        for chunk in pd.read_csv(z.open('studentVle.csv'),na_values=['?'],chunksize=500000):chunks.append(chunk.loc[chunk.date.between(0,56)])
    logs=pd.concat(chunks,ignore_index=True);d,audit=e1.features_from_logs(cohort,logs,sites,cutoff=56);d['enrolment_key']=key(d)
    d.to_csv(ROOT/'day56_features.csv',index=False);del chunks,logs;gc.collect()
    previous=pd.read_csv(PARENT/'E3_oulad_multihorizon_v1.0/day56_features.csv').set_index('enrolment_key').sort_index();now=d.set_index('enrolment_key').sort_index()
    pd.testing.assert_frame_equal(now[e1.FEATURES],previous[e1.FEATURES],check_dtype=False,atol=1e-10,rtol=1e-10)
    assert now.index.equals(previous.index) and np.array_equal(now.final_result,previous.final_result)
    assert audit['retained_log_max_day']==56 and len(d)==26412
    dump(ROOT/'raw_feature_audit.json',dict(raw_sha256=e1.sha(raw),records=len(d),students=d.id_student.nunique(),features=16,E3_features_numerically_identical=True,exclusion_counts=allrows.cohort_status.value_counts().to_dict(),**audit))
    shutil.copy2(A/'requirements-lock.txt',ROOT/'requirements-lock.txt')
    outer=[];inneraudit=[]
    for g,t in itertools.product(MOD,TASKS):
        held,tr,te,inner=engine.subsets(d,g,t);dest=ROOT/'folds'/g;dest.mkdir(parents=True,exist_ok=True);p=dest/t;p.mkdir(exist_ok=True)
        pd.DataFrame({'id_student':sorted(held)}).to_csv(dest/'heldout_students.csv',index=False)
        for role,z in [('train',tr),('test',te)]:
            pd.DataFrame({'id_student':sorted(z.id_student.unique())}).to_csv(p/f'{role}_students.csv',index=False)
            z[e1.KEY+['row_id','enrolment_key','final_result','y_'+t]].to_csv(p/f'{role}_membership.csv',index=False)
        outer.append(dict(heldout_module=g,task=t,train_records=len(tr),train_students=tr.id_student.nunique(),test_records=len(te),test_students=te.id_student.nunique(),train_test_student_overlap=len(set(tr.id_student)&held),test_positive=int(te['y_'+t].sum()),test_prevalence=te['y_'+t].mean(),train_positive=int(tr['y_'+t].sum())))
        for h,it,iv in inner:
            for role,z in [('train',it),('validation',iv)]:z[e1.KEY+['row_id']].to_csv(p/f'inner_{h}_{role}_membership.csv',index=False)
            inneraudit.append(dict(heldout_module=g,task=t,validation_module=h,train_records=len(it),validation_records=len(iv),train_positive=int(it['y_'+t].sum()),validation_positive=int(iv['y_'+t].sum()),train_validation_student_overlap=len(set(it.id_student)&set(iv.id_student)),inner_train_outer_test_student_overlap=len(set(it.id_student)&held),inner_validation_outer_test_student_overlap=len(set(iv.id_student)&held)))
    pd.DataFrame(outer).to_csv(ROOT/'outer_fold_audit.csv',index=False);pd.DataFrame(inneraudit).to_csv(ROOT/'inner_fold_audit.csv',index=False)
    fixed=['PROTOCOL.txt','run_e4b.py','e1_reference.py','e4a_engine.py','day56_features.csv','day56_cohort_audit.csv']
    dump(ROOT/'run_manifest.json',dict(seed=e1.SEED,landmark=56,families=NAMES,features=e1.FEATURES,studies=42,trials=630,inner_fits=3780,final_fits=42,fixed_hashes={f:e1.sha(ROOT/f) for f in fixed},split_hashes={str(p.relative_to(ROOT)).replace('\\','/'):e1.sha(p) for p in (ROOT/'folds').rglob('*.csv')}))
    print(pd.DataFrame(outer).to_string(index=False),flush=True)
def train(workers):
    integrity();frozen_check();assert load(ROOT/'pretraining_audit.json')['status']=='passed'
    def one(job):
        g,t,n=job;dest=ROOT/'folds'/g/t
        if (dest/n/'selection.json').exists():return job
        with (dest/f'{n}_train.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u',str(Path(__file__)),'worker','--module',g,'--task',t,'--model',n],stdout=f,stderr=subprocess.STDOUT,check=True)
        return job
    jobs=list(itertools.product(MOD,TASKS,NAMES))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,j) for j in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):print('DONE',i,'/42',f.result(),flush=True)
    selected=[]
    for g,t in itertools.product(MOD,TASKS):
        candidates=[load(ROOT/'folds'/g/t/n/'selection.json') for n in NAMES];best=max(candidates,key=lambda s:s['inner_mean_AP'])
        selected.append(dict(heldout_module=g,task=t,inner_selected_model=best['model'],inner_mean_AP=best['inner_mean_AP']))
    pd.DataFrame(selected).to_csv(ROOT/'inner_selected_models.csv',index=False)
    dump(ROOT/'training_complete.json',dict(status='passed',studies=42,trials=630,selection_sha256=e1.sha(ROOT/'inner_selected_models.csv')))
def score():
    integrity();frozen_check();assert e1.sha(ROOT/'inner_selected_models.csv')==load(ROOT/'training_complete.json')['selection_sha256']
    d=frame();choices=pd.read_csv(ROOT/'inner_selected_models.csv');rows=[]
    for g,t,n in itertools.product(MOD,TASKS,NAMES):
        dest=ROOT/'folds'/g/t/n;s=load(dest/'selection.json');obj=joblib.load(dest/'model.joblib');_,tr,te,_=engine.subsets(d,g,t)
        y=te['y_'+t].to_numpy(dtype=int);p=obj['model'].predict_proba(te[e1.FEATURES])[:,1]
        pred=te[e1.KEY+['row_id','enrolment_key','final_result']].copy();pred['y']=y;pred['probability']=p;pred.to_csv(dest/'test_predictions.csv',index=False)
        chosen=choices.loc[(choices.heldout_module==g)&(choices.task==t),'inner_selected_model'].iloc[0]
        lo,hi=e1.bootstrap_ap(y,p,te.id_student.to_numpy(),1000)
        rows.append(dict(heldout_module=g,task=t,model=n,inner_selected_model=chosen,is_validation_selected=n==chosen,inner_mean_AP=s['inner_mean_AP'],best_trial=s['best_trial'],completed_trials=15,**engine.evaluate(y,p,te.enrolment_key.to_numpy(),s['threshold']),AP_CI_low=lo,AP_CI_high=hi))
        del obj;gc.collect()
    r=pd.DataFrame(rows);r.to_csv(ROOT/'all_models_test_results.csv',index=False);selected=r.loc[r.is_validation_selected].copy();selected.to_csv(ROOT/'validation_selected_test_results.csv',index=False)
    allr=pd.concat([r,selected.assign(model='Validation-selected')],ignore_index=True);macro=[]
    for (t,n),z in allr.groupby(['task','model']):
        row=dict(task=t,model=n)
        for metric in ['AP','nAP','AUROC','Brier','Lift_at_10']:
            row.update({f'Mean_{metric}':z[metric].mean(),f'SD_{metric}':z[metric].std(ddof=0),f'Min_{metric}':z[metric].min(),f'Max_{metric}':z[metric].max()})
        macro.append(row)
    pd.DataFrame(macro).to_csv(ROOT/'macro_domain_summary.csv',index=False)
    print(pd.DataFrame(macro)[['task','model','Mean_AP','Mean_nAP','Min_nAP','Mean_AUROC']].to_string(index=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','worker','score']);p.add_argument('--raw');p.add_argument('--workers',type=int,default=3);p.add_argument('--module');p.add_argument('--task');p.add_argument('--model');a=p.parse_args()
    if a.stage=='prepare':prepare(a.raw)
    elif a.stage=='train':train(a.workers)
    elif a.stage=='worker':engine.worker(a.module,a.task,a.model)
    else:score()
