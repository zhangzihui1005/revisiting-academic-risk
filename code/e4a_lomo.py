"""E4A frozen nested, student-disjoint leave-one-module-out experiment.
Run prepare once, audit pretraining, train, then summarize and audit final.
Dependencies and model/search-space implementations are copied from frozen E1.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
import sys
sys.dont_write_bytecode=True
import argparse, json, shutil, time, subprocess, concurrent.futures, itertools, gc
from pathlib import Path
import numpy as np
import pandas as pd
import optuna, joblib
from sklearn.metrics import average_precision_score, f1_score
import e1_reference as e1

ROOT=Path(__file__).resolve().parent
MODULES=list('ABCDEFG')
MODULES=[s*3 for s in MODULES]
TASKS=['withdrawal','failure']
NAMES=['LR','RF','XGBoost','LightGBM','CatBoost']
TRIALS=15
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,o):e1.dump(p,o)
def frame():
    d=pd.read_csv(ROOT/'day28_features.csv')
    d['enrolment_key']=d.code_module+'|'+d.code_presentation+'|'+d.id_student.astype(str)
    return d
def subsets(d,g,task):
    held=set(d.loc[d.code_module.eq(g),'id_student'])
    source=d.loc[~d.code_module.eq(g)&~d.id_student.isin(held)]
    train=source.loc[source['y_'+task].notna()]
    test=d.loc[d.code_module.eq(g)&d['y_'+task].notna()]
    inner=[]
    for h in MODULES:
        if h==g:continue
        vi=set(source.loc[source.code_module.eq(h),'id_student'])
        va=train.loc[train.code_module.eq(h)]
        tr=train.loc[~train.code_module.eq(h)&~train.id_student.isin(vi)]
        inner.append((h,tr,va))
    return held,train,test,inner
def capacity(y,p,keys):
    y=np.asarray(y,dtype=int);p=np.asarray(p);prev=y.mean();n=len(y)
    order=np.lexsort((np.asarray(keys,dtype=str),-p));out={}
    for pct in [5,10,20]:
        k=int(np.ceil(n*pct/100));hit=int(y[order[:k]].sum());prec=hit/k
        out.update({f'Precision_at_{pct}':prec,f'Recall_at_{pct}':hit/y.sum(),f'Lift_at_{pct}':prec/prev,f'selected_N_at_{pct}':k,f'hits_at_{pct}':hit})
    return out
def evaluate(y,p,keys,threshold):
    m=e1.metrics(y,p,threshold);ap=m.pop('AUPRC');prev=np.mean(y)
    return dict(AP=ap,nAP=(ap-prev)/(1-prev),test_N=len(y),test_Pos=int(np.sum(y)),test_Prevalence=prev,**m,F1_at_0_5=f1_score(y,p>=.5,zero_division=0),**capacity(y,p,keys))
def threshold_curve(oof):
    # Exact candidate union. Mean of six domain F1s, not pooled F1.
    cand=np.unique(np.r_[0.,oof.probability.to_numpy(),1.])
    vals=[]
    for _,s in oof.groupby('code_module',sort=True):
        order=np.argsort(s.probability.to_numpy(),kind='stable')
        p=s.probability.to_numpy()[order];y=s.y.to_numpy()[order]
        pos=np.searchsorted(p,cand,side='left')
        tp=np.r_[0,np.cumsum(y)][-1]-np.r_[0,np.cumsum(y)][pos]
        vals.append(2*tp/np.maximum(y.sum()+len(y)-pos,1))
    mean=np.mean(vals,axis=0)
    c=pd.DataFrame({'threshold':cand,'mean_domain_F1':mean})
    return float(cand[np.argmax(mean)]),c
def frozen_check():
    f=load(ROOT/'frozen_sources.json');n=0
    for group in f.values():
        base=Path(group['directory'])
        if not base.exists():raise FileNotFoundError(base)
        for rel,h in group['files'].items():
            assert e1.sha(base/rel)==h,(base,rel)
            n+=1
    return n
def prepare():
    assert not (ROOT/'run_manifest.json').exists(),'Already prepared; do not overwrite frozen protocol.'
    parent=ROOT.parent
    frozen={}
    for label,dirname in [('E1','experiment1_day28'),('E2','E2_day28_formulation_v1.0'),('E3','E3_oulad_multihorizon_v1.0')]:
        base=parent/dirname
        frozen[label]={'directory':str(base),'files':{str(p.relative_to(base)).replace('\\','/'):e1.sha(p) for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
    dump(ROOT/'frozen_sources.json',frozen)
    shutil.copy2(parent/'experiment1_day28/oulad_day28_features.csv',ROOT/'day28_features.csv')
    shutil.copy2(parent/'experiment1_day28/requirements-lock.txt',ROOT/'requirements-lock.txt')
    for task in TASKS:
        shutil.copy2(parent/f'experiment1_day28/oulad_day28_{task}_results.csv',ROOT/f'E1_{task}_reference.csv')
        shutil.copy2(parent/f'experiment1_day28/{task}_validation_selected_model.json',ROOT/f'E1_{task}_selected.json')
    d=frame();outeraudit=[];inneraudit=[]
    for g in MODULES:
        dest=ROOT/'folds'/g;dest.mkdir(parents=True,exist_ok=True)
        held=sorted(d.loc[d.code_module.eq(g),'id_student'].unique())
        pd.DataFrame({'id_student':held}).to_csv(dest/'heldout_students.csv',index=False)
        for task in TASKS:
            held,tr,te,inner=subsets(d,g,task);p=dest/task;p.mkdir(exist_ok=True)
            for name,z in [('train',tr),('test',te)]:
                pd.DataFrame({'id_student':sorted(z.id_student.unique())}).to_csv(p/f'{name}_students.csv',index=False)
                z[e1.KEY+['row_id','enrolment_key','final_result','y_'+task]].to_csv(p/f'{name}_membership.csv',index=False)
            overlap=len(set(tr.id_student)&set(te.id_student))
            ou=dict(heldout_module=g,task=task,train_records=len(tr),train_students=tr.id_student.nunique(),test_records=len(te),test_students=te.id_student.nunique(),train_test_student_overlap=overlap,test_positive=int(te['y_'+task].sum()),test_prevalence=te['y_'+task].mean(),source_records_removed_for_outer_student_overlap=int((~d.code_module.eq(g)&d['y_'+task].notna()&d.id_student.isin(held)).sum()))
            ou['train_positive']=int(tr['y_'+task].sum());outeraudit.append(ou)
            for h,it,iv in inner:
                for role,z in [('train',it),('validation',iv)]:
                    z[e1.KEY+['row_id']].to_csv(p/f'inner_{h}_{role}_membership.csv',index=False)
                inneraudit.append(dict(heldout_module=g,task=task,validation_module=h,train_records=len(it),train_students=it.id_student.nunique(),validation_records=len(iv),validation_students=iv.id_student.nunique(),train_positive=int(it['y_'+task].sum()),validation_positive=int(iv['y_'+task].sum()),validation_prevalence=iv['y_'+task].mean(),train_validation_student_overlap=len(set(it.id_student)&set(iv.id_student)),inner_train_outer_test_student_overlap=len(set(it.id_student)&held),inner_validation_outer_test_student_overlap=len(set(iv.id_student)&held)))
    pd.DataFrame(outeraudit).to_csv(ROOT/'outer_fold_audit.csv',index=False)
    pd.DataFrame(inneraudit).to_csv(ROOT/'inner_fold_audit.csv',index=False)
    dump(ROOT/'run_manifest.json',dict(seed=e1.SEED,trials_per_study=TRIALS,studies=70,inner_fits=6300,final_fits=70,features=e1.FEATURES,protocol_sha256=e1.sha(ROOT/'PROTOCOL.txt'),feature_sha256=e1.sha(ROOT/'day28_features.csv'),reference_code_sha256=e1.sha(ROOT/'e1_reference.py'),training_code_sha256=e1.sha(Path(__file__)),split_hashes={str(p.relative_to(ROOT)).replace('\\','/'):e1.sha(p) for p in (ROOT/'folds').rglob('*.csv')},created=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(pd.DataFrame(outeraudit).to_string(index=False),flush=True)
def integrity():
    m=load(ROOT/'run_manifest.json')
    for name,key in [('PROTOCOL.txt','protocol_sha256'),('day28_features.csv','feature_sha256'),('e1_reference.py','reference_code_sha256'),('run_e4a.py','training_code_sha256')]:assert e1.sha(ROOT/name)==m[key],name
    for name,h in m['split_hashes'].items():assert e1.sha(ROOT/name)==h,name
def worker(g,task,name):
    integrity();assert load(ROOT/'pretraining_audit.json')['status']=='passed'
    dest=ROOT/'folds'/g/task/name;dest.mkdir(exist_ok=True)
    if (dest/'selection.json').exists():return
    start=time.time();d=frame();_,tr,te,inner=subsets(d,g,task)
    # Trials never receive outer test rows or labels.
    parts=[(h,a[e1.FEATURES],a['y_'+task].to_numpy(dtype=int),b[e1.FEATURES],b['y_'+task].to_numpy(dtype=int),b) for h,a,b in inner]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=e1.SEED),storage='sqlite:///'+str((dest/'study.sqlite3').resolve()).replace('\\','/'),study_name=f'{g}_{task}_{name}',load_if_exists=False)
    best=-np.inf
    def objective(trial):
        nonlocal best
        params=e1.parameters(trial,name);aps=[];naps=[];oof=[];domain=[]
        for h,x,y,vx,vy,meta in parts:
            m=e1.model(name,params);m.fit(x,y);p=m.predict_proba(vx)[:,1]
            ap=average_precision_score(vy,p);prev=vy.mean();nap=(ap-prev)/(1-prev)
            aps.append(ap);naps.append(nap)
            domain.append(dict(validation_module=h,AP=ap,nAP=nap,prevalence=prev,N=len(vy)))
            o=meta[e1.KEY+['row_id','enrolment_key']].copy();o['y']=vy;o['probability']=p;oof.append(o)
            del m
        val=float(np.mean(aps))
        trial.set_user_attr('domains',domain)
        trial.set_user_attr('SD_AP',float(np.std(aps,ddof=0)))
        trial.set_user_attr('Worst_AP',float(np.min(aps)))
        trial.set_user_attr('Mean_nAP',float(np.mean(naps)))
        if val>best:
            best=val
            pd.concat(oof,ignore_index=True).to_csv(dest/'best_inner_predictions.csv.gz',index=False)
            dump(dest/'best_inner_prediction_trial.json',dict(trial=trial.number,mean_AP=val))
        print(f'{g} {task} {name} trial={trial.number+1}/15 mean_AP={val:.6f}',flush=True)
        return val
    study.optimize(objective,n_trials=TRIALS)
    study.trials_dataframe().to_csv(dest/'trials.csv',index=False)
    rows=[]
    for t in study.trials:
        for item in t.user_attrs['domains']:rows.append(dict(trial=t.number,mean_AP=t.value,SD_AP=t.user_attrs['SD_AP'],Worst_AP=t.user_attrs['Worst_AP'],Mean_nAP=t.user_attrs['Mean_nAP'],**item))
    pd.DataFrame(rows).to_csv(dest/'trial_inner_domains.csv',index=False)
    bt=study.best_trial
    assert load(dest/'best_inner_prediction_trial.json')['trial']==bt.number
    oof=pd.read_csv(dest/'best_inner_predictions.csv.gz');threshold,curve=threshold_curve(oof)
    curve.to_csv(dest/'inner_threshold_curve.csv.gz',index=False)
    m=e1.model(name,bt.params);m.fit(tr[e1.FEATURES],tr['y_'+task].astype(int))
    joblib.dump(dict(model=m,features=e1.FEATURES,threshold=threshold,seed=e1.SEED,heldout_module=g,task=task,family=name,best_trial=bt.number),dest/'model.joblib',compress=3)
    # Outer test evaluation is a later, separate command, after family selection.
    dump(dest/'selection.json',dict(heldout_module=g,task=task,model=name,best_trial=bt.number,params=bt.params,inner_mean_AP=bt.value,inner_SD_AP=bt.user_attrs['SD_AP'],inner_Worst_AP=bt.user_attrs['Worst_AP'],inner_Mean_nAP=bt.user_attrs['Mean_nAP'],threshold=threshold,threshold_mean_domain_F1=float(curve.mean_domain_F1.max()),completed_trials=len(study.trials),runtime_seconds=time.time()-start,model_sha256=e1.sha(dest/'model.joblib')))
    print('FINISHED',g,task,name,flush=True)
def train(workers):
    integrity();assert load(ROOT/'pretraining_audit.json')['status']=='passed'
    frozen_check();jobs=list(itertools.product(MODULES,TASKS,NAMES))
    def one(job):
        g,t,n=job;log=ROOT/'folds'/g/t/f'{n}_train.log'
        with log.open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u',str(Path(__file__)),'worker','--module',g,'--task',t,'--model',n],stdout=f,stderr=subprocess.STDOUT,check=True)
        return job
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fut={pool.submit(one,j):j for j in jobs}
        for i,f in enumerate(concurrent.futures.as_completed(fut),1):
            print(f'DONE {i}/70 {f.result()}',flush=True)
    selected=[]
    for g,t in itertools.product(MODULES,TASKS):
        candidates=[load(ROOT/'folds'/g/t/n/'selection.json') for n in NAMES]
        choice=max(candidates,key=lambda r:r['inner_mean_AP'])
        selected.append(dict(heldout_module=g,task=t,inner_selected_model=choice['model'],inner_mean_AP=choice['inner_mean_AP']))
    pd.DataFrame(selected).to_csv(ROOT/'inner_selected_models.csv',index=False)
    dump(ROOT/'training_complete.json',dict(status='passed',studies=70,completed_trials=1050,inner_fits=6300,final_fits=70,selection_sha256=e1.sha(ROOT/'inner_selected_models.csv')))
def summarize():
    integrity();frozen_check();done=load(ROOT/'training_complete.json')
    assert e1.sha(ROOT/'inner_selected_models.csv')==done['selection_sha256']
    d=frame();rows=[];student=[];sel=pd.read_csv(ROOT/'inner_selected_models.csv')
    for g,task,name in itertools.product(MODULES,TASKS,NAMES):
        dest=ROOT/'folds'/g/task/name;s=load(dest/'selection.json');obj=joblib.load(dest/'model.joblib')
        _,tr,te,_=subsets(d,g,task);y=te['y_'+task].to_numpy(dtype=int);p=obj['model'].predict_proba(te[e1.FEATURES])[:,1]
        pred=te[e1.KEY+['row_id','enrolment_key','final_result']].copy();pred['y']=y;pred['probability']=p
        pred.to_csv(dest/'test_predictions.csv',index=False)
        lo,hi=e1.bootstrap_ap(y,p,te.id_student.to_numpy(),repeats=1000)
        chosen=sel.loc[sel.heldout_module.eq(g)&sel.task.eq(task),'inner_selected_model'].iloc[0]
        rows.append(dict(heldout_module=g,task=task,model=name,inner_selected_model=chosen,is_validation_selected=name==chosen,**{k:s[k] for k in ['best_trial','inner_mean_AP','inner_SD_AP','inner_Worst_AP','inner_Mean_nAP','completed_trials','runtime_seconds']},**evaluate(y,p,te.enrolment_key.to_numpy(),s['threshold']),AP_CI_low=lo,AP_CI_high=hi))
        if task=='withdrawal':
            st=pred.groupby('id_student',sort=True).agg(y=('y','max'),probability=('probability','max')).reset_index()
            student.append(dict(heldout_module=g,model=name,is_validation_selected=name==chosen,students=len(st),positive_students=int(st.y.sum()),prevalence=st.y.mean(),**capacity(st.y.to_numpy(),st.probability.to_numpy(),st.id_student.astype(str).str.zfill(12).to_numpy())))
        del obj;gc.collect()
    allr=pd.DataFrame(rows);allr.to_csv(ROOT/'all_models_test_results.csv',index=False)
    chosen=allr.loc[allr.is_validation_selected].copy();chosen.to_csv(ROOT/'validation_selected_test_results.csv',index=False)
    pd.DataFrame(student).to_csv(ROOT/'withdrawal_student_level_capacity.csv',index=False)
    combined=pd.concat([allr,chosen.assign(model='Validation-selected')],ignore_index=True)
    macro=[];gaps=[];rankstats=[];reversals=[]
    for task in TASKS:
        r=allr.loc[allr.task.eq(task)]
        ap=r.pivot(index='heldout_module',columns='model',values='AP').reindex(index=MODULES,columns=NAMES)
        nap=r.pivot(index='heldout_module',columns='model',values='nAP').reindex(index=MODULES,columns=NAMES)
        ap.to_csv(ROOT/f'{task}_model_module_AP.csv');nap.to_csv(ROOT/f'{task}_model_module_nAP.csv')
        ranks=ap.rank(axis=1,ascending=False,method='average');ranks.to_csv(ROOT/f'{task}_model_module_ranks.csv')
        for n in NAMES:rankstats.append(dict(task=task,model=n,mean_rank=ranks[n].mean(),SD_rank=ranks[n].std(ddof=0),best_rank=ranks[n].min(),worst_rank=ranks[n].max(),first_place_count=int(ranks[n].eq(1).sum())))
        for a,b in itertools.combinations(NAMES,2):
            delta=ap[a]-ap[b]
            reversals.append(dict(task=task,model_a=a,model_b=b,a_higher=int((delta>1e-12).sum()),b_higher=int((delta<-1e-12).sum()),has_reversal=bool(delta.min() < -1e-12 and delta.max() > 1e-12),min_AP_difference=delta.min(),max_AP_difference=delta.max()))
        ref=pd.read_csv(ROOT/f'E1_{task}_reference.csv').set_index('model')
        for n in NAMES+['Validation-selected']:
            s=combined.loc[combined.task.eq(task)&combined.model.eq(n)]
            row=dict(task=task,model=n,domains=len(s))
            for metric in ['AP','nAP','AUROC','Brier','Lift_at_10','Precision_at_10','Recall_at_10','F1']:
                row.update({f'Mean_{metric}':s[metric].mean(),f'SD_{metric}':s[metric].std(ddof=0),f'Min_{metric}':s[metric].min(),f'Max_{metric}':s[metric].max()})
            row.update(Worst_AP_module=s.loc[s.AP.idxmin(),'heldout_module'],Worst_nAP_module=s.loc[s.nAP.idxmin(),'heldout_module'])
            macro.append(row)
            refname=load(ROOT/f'E1_{task}_selected.json')['model'] if n=='Validation-selected' else n
            ri=ref.loc[refname];prev=ri.test_prevalence;ina=(ri.AUPRC-prev)/(1-prev)
            gaps.append(dict(task=task,model=n,E1_reference_model=refname,E1_AP=ri.AUPRC,E1_nAP=ina,E1_AUROC=ri.AUROC,LOMO_Mean_AP=row['Mean_AP'],LOMO_Mean_nAP=row['Mean_nAP'],LOMO_Mean_AUROC=row['Mean_AUROC'],Gap_AP=ri.AUPRC-row['Mean_AP'],Gap_nAP=ina-row['Mean_nAP'],Gap_AUROC=ri.AUROC-row['Mean_AUROC']))
    pd.DataFrame(macro).to_csv(ROOT/'macro_domain_summary.csv',index=False)
    pd.DataFrame(gaps).to_csv(ROOT/'generalization_gaps.csv',index=False)
    pd.DataFrame(rankstats).to_csv(ROOT/'rank_stability.csv',index=False)
    pd.DataFrame(reversals).to_csv(ROOT/'pairwise_rank_reversals.csv',index=False)
    dump(ROOT/'summary_checks.json',dict(status='passed',test_result_rows=len(allr),validation_selected_rows=len(chosen),frozen_files_verified=frozen_check(),outer_evaluation_started_after_family_selection=True))
    print(pd.DataFrame(macro)[['task','model','Mean_AP','Mean_nAP','Min_nAP','Mean_AUROC','Mean_Lift_at_10']].to_string(index=False),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','worker','train','summarize']);a.add_argument('--module');a.add_argument('--task');a.add_argument('--model');a.add_argument('--workers',type=int,default=3);args=a.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='worker':worker(args.module,args.task,args.model)
    elif args.stage=='train':train(args.workers)
    else:summarize()
