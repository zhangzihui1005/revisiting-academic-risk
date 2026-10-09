"""Frozen-feature multihorizon study. Stages: prepare -> independent audit -> train -> summarize."""
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
import argparse,json,time,zipfile,shutil,gc,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd,optuna,joblib
from sklearn.metrics import average_precision_score,f1_score
import e1_reference as e1

HORIZONS=[14,28,56,84];NAMES=['LR','RF','XGBoost','LightGBM','CatBoost'];TASKS=['withdrawal','failure']
SEED=e1.SEED;FEATURES=e1.FEATURES;KEY=e1.KEY
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,o):e1.dump(p,o)
def key(df):return df.code_module.astype(str)+'|'+df.code_presentation.astype(str)+'|'+df.id_student.astype(str)
def frame(out,t):
    d=pd.read_csv(out/f'day{t}_features.csv');d['enrolment_key']=key(d);return d

def frozen_check(out):
    for item in load(out/'frozen_sources.json').values():
        for f,h in item['files'].items():assert e1.sha(Path(item['directory'])/f)==h,f
        if 'snapshot_directory' in item:
            for f,h in item['files'].items():assert e1.sha(Path(item['snapshot_directory'])/f)==h,f

def temporal_tests():
    c=pd.DataFrame([dict(code_module='A',code_presentation='P',id_student=1,row_id=0,date_registration=-2,
        final_result='Pass',y_withdrawal=0,y_failure=0,num_of_prev_attempts=0,studied_credits=60)])
    sites=pd.DataFrame([dict(id_site=1,code_module='A',code_presentation='P',activity_type='resource')])
    for t in HORIZONS:
        logs=pd.DataFrame([dict(code_module='A',code_presentation='P',id_student=1,id_site=1,date=d,sum_click=v) for d,v in [(0,2),(t-7,4),(t,3),(t+1,99999),(-1,88888)]])
        a,_=e1.features_from_logs(c,logs,sites,cutoff=t)
        b,_=e1.features_from_logs(c,logs.loc[logs.date.between(0,t)],sites,cutoff=t)
        pd.testing.assert_frame_equal(a,b)
        assert a.total_clicks.iloc[0]==9 and a.recent7_clicks.iloc[0]==3 and a.previous7_clicks.iloc[0]==4
        assert np.isclose(a.active_ratio.iloc[0],3/(t+1))

def prepare(oulad,base,out):
    frozen_check(out);temporal_tests()
    assert e1.sha(out/'e1_reference.py')==e1.sha(base/'run_experiment.py')
    original_manifest=load(base/'run_manifest.json')
    assert e1.sha(oulad) in original_manifest['inputs'].values()
    for p,v in original_manifest['versions'].items():assert importlib.metadata.version(p)==v
    old=pd.read_csv(base/'oulad_day28_features.csv')
    with zipfile.ZipFile(oulad) as z:info=e1.read(z,'studentInfo.csv');reg=e1.read(z,'studentRegistration.csv');sites=e1.read(z,'vle.csv')
    assert old.groupby('id_student').split.nunique().max()==1
    known=old[['id_student','split']].drop_duplicates().sort_values('id_student')
    missing=np.array(sorted(set(info.id_student)-set(known.id_student)),dtype=int)
    newids=np.random.default_rng(SEED).permutation(missing);n=len(newids);ntrain=int(.6*n);nval=int(.2*n)
    extra=pd.DataFrame({'id_student':newids,'split':['train']*ntrain+['validation']*nval+['test']*(n-ntrain-nval),'assignment_source':'new_seeded'})
    master=pd.concat([known.assign(assignment_source='E1_frozen'),extra],ignore_index=True).sort_values('id_student')
    assert len(master)==info.id_student.nunique() and not master.id_student.duplicated().any()
    masterpath=out/'master_student_split.csv'
    if masterpath.exists():pd.testing.assert_frame_equal(pd.read_csv(masterpath),master.reset_index(drop=True))
    else:master.to_csv(masterpath,index=False)
    mapping=master.set_index('id_student').split
    assert np.array_equal(old.id_student.map(mapping),old.split)
    e2path=Path(load(out/'frozen_sources.json')['E2']['directory'])
    e2split=pd.read_csv(e2path/'frozen_labels_and_split.csv')
    assert np.array_equal(e2split.id_student.map(mapping),e2split.split)
    auditlist=[];cohortsummary=[];splits=[]
    for t in HORIZONS:
        print('PREPARE',t,flush=True)
        cohort,allrecords=e1.active_cohort(info,reg,cutoff=t)
        allrecords[KEY+['date_registration','date_unregistration','final_result','cohort_status']].to_csv(out/f'day{t}_cohort_audit.csv',index=False)
        assert cohort.date_registration.le(t).all()
        assert not cohort.date_unregistration.le(t).any()
        assert cohort.loc[cohort.final_result.eq('Withdrawn'),'date_unregistration'].gt(t).all()
        if t==28:
            shutil.copy2(base/'oulad_day28_features.csv',out/'day28_features.csv')
            d=frame(out,28)
            assert list(key(cohort))==list(d.enrolment_key)
            logaudit={k:v for k,v in load(base/'data_audit.json').items() if k.startswith('retained_log') or k=='zero_activity_records'}
            assert e1.sha(out/'day28_features.csv')==e1.sha(base/'oulad_day28_features.csv')
        else:
            chunks=[]
            with zipfile.ZipFile(oulad) as z:
                for chunk in pd.read_csv(z.open('studentVle.csv'),na_values=['?'],chunksize=500000):chunks.append(chunk.loc[chunk.date.between(0,t)])
            logs=pd.concat(chunks,ignore_index=True)
            d,logaudit=e1.features_from_logs(cohort,logs,sites,cutoff=t)
            d['split']=d.id_student.map(mapping);d['enrolment_key']=key(d)
            d.to_csv(out/f'day{t}_features.csv',index=False)
            del chunks,logs;gc.collect()
        assert np.array_equal(d.id_student.map(mapping),d.split)
        assert logaudit['retained_log_max_day']<=t
        row=dict(landmark=t,records=len(d),students=d.id_student.nunique(),exclusion_counts=allrecords.cohort_status.value_counts().to_dict(),**logaudit)
        auditlist.append(row);print('COHORT',json.dumps(row,default=int),flush=True)
        for task in TASKS:
            taskdf=d.loc[d['y_'+task].notna()]
            cohortsummary.append(dict(landmark=t,task=task,cohort_N=len(taskdf),cohort_students=taskdf.id_student.nunique(),cohort_Pos=int(taskdf['y_'+task].sum()),cohort_Prevalence=taskdf['y_'+task].mean()))
            for split,part in taskdf.groupby('split'):
                splits.append(dict(landmark=t,task=task,split=split,N=len(part),students=part.id_student.nunique(),Pos=int(part['y_'+task].sum()),Prevalence=part['y_'+task].mean()))
    sets=[set(frame(out,t).enrolment_key) for t in HORIZONS]
    common=set.intersection(*sets);d84=frame(out,84)
    stable=d84.loc[d84.enrolment_key.isin(common),KEY+['enrolment_key','split','final_result','y_withdrawal','y_failure']].copy()
    a84=pd.read_csv(out/'day84_cohort_audit.csv');a84['enrolment_key']=key(a84)
    expected=set(a84.loc[a84.cohort_status.eq('included')&a84.date_registration.le(14),'enrolment_key'])
    assert common==expected
    stable.to_csv(out/'stable_cohort_membership.csv',index=False)
    dump(out/'stable_cohort_definition.json',{'definition':'intersection of all four active-enrolment risk sets; equals R84 with registration<=14',
        'R84_records':len(d84),'common_records':len(stable),'removed_from_R84':len(d84)-len(stable),
        'common_test_records':int(stable.split.eq('test').sum()),'evaluation_only':True,'retrospective_conditioning':True})
    pd.DataFrame(cohortsummary).to_csv(out/'cohort_summary.csv',index=False)
    pd.DataFrame(splits).to_csv(out/'split_summary.csv',index=False)
    dump(out/'preparation_audit.json',{'horizons':auditlist,'master_students':len(master),'E1_existing_students':len(known),'new_assigned_students':n,
        'existing_assignments_unchanged':True,'feature_code_identical_to_E1':True,'temporal_tests':'passed'})
    dump(out/'run_manifest.json',{'seed':SEED,'horizons':HORIZONS,'new_horizons':[14,56,84],'features':FEATURES,'versions':original_manifest['versions'],
        'raw_oulad_sha256':e1.sha(oulad),'master_split_sha256':e1.sha(masterpath),'feature_hashes':{str(t):e1.sha(out/f'day{t}_features.csv') for t in HORIZONS},
        'protocol_sha256':e1.sha(out/'PROTOCOL.txt'),'new_trials_planned':900,'Day28_retrained':False})
    print('PREPARATION_COMPLETE',flush=True)

def capacity(y,p,keys):
    y=np.asarray(y,dtype=int);p=np.asarray(p);prev=y.mean();n=len(y)
    order=np.lexsort((np.asarray(keys,dtype=str),-p));out={}
    for pct in [5,10,20]:
        k=int(np.ceil(n*pct/100));hit=int(y[order[:k]].sum());prec=hit/k;rec=hit/y.sum()
        out.update({f'Precision_at_{pct}':prec,f'Recall_at_{pct}':rec,f'Lift_at_{pct}':prec/prev,f'selected_N_at_{pct}':k,f'hits_at_{pct}':hit})
    return out

def evaluate(y,p,keys,threshold):
    m=e1.metrics(y,p,threshold);ap=m.pop('AUPRC');prev=np.mean(y)
    return dict(AP=ap,nAP=(ap-prev)/(1-prev),test_N=len(y),test_Pos=int(np.sum(y)),test_Prevalence=prev,
        **m,F1_at_0_5=f1_score(y,p>=.5,zero_division=0),**capacity(y,p,keys))

def train_new(base,out):
    assert load(out/'pretraining_independent_audit.json')['status']=='passed'
    manifest=load(out/'run_manifest.json')
    assert e1.sha(out/'master_student_split.csv')==manifest['master_split_sha256']
    assert e1.sha(out/'PROTOCOL.txt')==manifest['protocol_sha256']
    for t,h in manifest['feature_hashes'].items():assert e1.sha(out/f'day{t}_features.csv')==h
    frozen_check(out);optuna.logging.set_verbosity(optuna.logging.WARNING)
    for t in [14,56,84]:
        d=frame(out,t);dest=out/f'day{t}';dest.mkdir(exist_ok=True);(dest/'models').mkdir(exist_ok=True);(dest/'trials').mkdir(exist_ok=True)
        for task in TASKS:
            taskdf=d.loc[d['y_'+task].notna()]
            parts={s:taskdf.loc[taskdf.split.eq(s)] for s in ['train','validation','test']}
            x={s:p[FEATURES] for s,p in parts.items()};y={s:p['y_'+task].to_numpy(dtype=int) for s,p in parts.items()}
            predictions={s:p[KEY+['row_id','enrolment_key','final_result','y_'+task]].copy() for s,p in parts.items() if s!='train'}
            results=[]
            for name in NAMES:
                print(f'START day{t} {task} {name}',flush=True);start=time.time()
                study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED),storage='sqlite:///'+str((out/'optuna_e3.sqlite3').resolve()).replace('\\','/'),study_name=f'day{t}_{task}_{name}',load_if_exists=True)
                fingerprint=manifest['feature_hashes'][str(t)]
                if study.user_attrs:assert study.user_attrs['feature_sha256']==fingerprint
                else:study.set_user_attr('feature_sha256',fingerprint)
                def objective(trial):
                    p=e1.parameters(trial,name);m=e1.model(name,p);ts=time.time();m.fit(x['train'],y['train'])
                    ap=average_precision_score(y['validation'],m.predict_proba(x['validation'])[:,1]);trial.set_user_attr('fit_seconds',time.time()-ts)
                    if (trial.number+1)%10==0:print(f'  day{t} {task} {name} trial {trial.number+1}/30 AP={ap:.4f}',flush=True)
                    return ap
                done=sum(r.state==optuna.trial.TrialState.COMPLETE for r in study.trials)
                study.optimize(objective,n_trials=max(0,30-done))
                m=e1.model(name,study.best_params);m.fit(x['train'],y['train'])
                vp=m.predict_proba(x['validation'])[:,1];tp=m.predict_proba(x['test'])[:,1]
                assert np.isclose(average_precision_score(y['validation'],vp),study.best_value)
                threshold=e1.threshold_from_validation(y['validation'],vp)
                lo,hi=e1.bootstrap_ap(y['test'],tp,parts['test'].id_student.to_numpy())
                metrics=evaluate(y['test'],tp,parts['test'].enrolment_key.to_numpy(),threshold)
                row=dict(landmark=t,task=task,model=name,validation_AP=study.best_value,**metrics,AP_CI_low=lo,AP_CI_high=hi,
                    n_train=len(y['train']),n_validation=len(y['validation']),completed_new_trials=sum(r.state==optuna.trial.TrialState.COMPLETE for r in study.trials),runtime_seconds=time.time()-start,source='new_E3_fit')
                results.append(row)
                for s,p in [('validation',vp),('test',tp)]:
                    predictions[s][name+'_probability']=p;predictions[s].to_csv(dest/f'{task}_{s}_predictions.csv',index=False)
                pd.DataFrame(results).to_csv(dest/f'{task}_results.csv',index=False)
                joblib.dump({'model':m,'features':FEATURES,'threshold':threshold,'seed':SEED},dest/'models'/f'{task}_{name}.joblib')
                dump(dest/'models'/f'{task}_{name}_parameters.json',study.best_params)
                study.trials_dataframe().to_csv(dest/'trials'/f'{task}_{name}.csv',index=False)
                print('RESULT',t,task,name,f'AP={metrics["AP"]:.4f} nAP={metrics["nAP"]:.4f}',flush=True)
            chosen=max(results,key=lambda r:r['validation_AP'])['model']
            dump(dest/f'{task}_selection.json',{'model':chosen,'criterion':'validation AP only'})
    print('NEW_TRAINING_COMPLETE',flush=True)

def reuse_day28(base,out):
    dest=out/'day28';dest.mkdir(exist_ok=True);d=frame(out,28)
    for task in TASKS:
        old=pd.read_csv(base/f'oulad_day28_{task}_results.csv');pred=pd.read_csv(base/f'{task}_test_predictions.csv');pred['enrolment_key']=key(pred)
        test=d.loc[d.split.eq('test')&d['y_'+task].notna()]
        assert np.array_equal(pred.row_id,test.row_id)
        rows=[];y=pred['y_'+task].to_numpy(dtype=int)
        for _,r in old.iterrows():
            m=evaluate(y,pred[r.model+'_probability'].to_numpy(),pred.enrolment_key.to_numpy(),r.threshold)
            assert np.isclose(m['AP'],r.AUPRC) and np.isclose(m['AUROC'],r.AUROC)
            rows.append(dict(landmark=28,task=task,model=r.model,validation_AP=r.validation_AUPRC,**m,AP_CI_low=r.AUPRC_CI_low,AP_CI_high=r.AUPRC_CI_high,
                n_train=r.n_train,n_validation=r.n_validation,completed_new_trials=0,runtime_seconds=0,source='frozen_E1_predictions'))
        pd.DataFrame(rows).to_csv(dest/f'{task}_results.csv',index=False)
        pred.to_csv(dest/f'{task}_test_predictions.csv',index=False)
        dump(dest/f'{task}_selection.json',load(base/f'{task}_validation_selected_model.json'))

def summarize(base,out):
    reuse_day28(base,out)
    allresults=[];selected=[];stable_results=[];student_top=[]
    stable=pd.read_csv(out/'stable_cohort_membership.csv');stablekeys=set(stable.loc[stable.split.eq('test'),'enrolment_key'])
    for t in HORIZONS:
        dest=out/f'day{t}'
        for task in TASKS:
            results=pd.read_csv(dest/f'{task}_results.csv');assert len(results)==5
            chosen=load(dest/f'{task}_selection.json')['model'];results['selected_on_validation']=results.model.eq(chosen)
            allresults.append(results);selected.append(results.loc[results.selected_on_validation])
            pred=pd.read_csv(dest/f'{task}_test_predictions.csv');common=pred.loc[pred.enrolment_key.isin(stablekeys)].copy()
            common.to_csv(dest/f'{task}_stable_test_predictions.csv',index=False)
            for _,r in results.iterrows():
                p=common[r.model+'_probability'].to_numpy();y=common['y_'+task].to_numpy(dtype=int)
                row=dict(landmark=t,task=task,model=r.model,selected_on_validation=bool(r.selected_on_validation),**evaluate(y,p,common.enrolment_key.to_numpy(),r.threshold))
                if r.selected_on_validation:
                    lo,hi=e1.bootstrap_ap(y,p,common.id_student.to_numpy());row.update(AP_CI_low=lo,AP_CI_high=hi)
                stable_results.append(row)
                if task=='withdrawal':
                    g=pred.groupby('id_student').agg(y=('y_withdrawal','max'),probability=(r.model+'_probability','max')).reset_index()
                    # numeric student id tie-break via zero-padded decimal representation
                    student_top.append(dict(landmark=t,model=r.model,selected_on_validation=bool(r.selected_on_validation),N_students=len(g),Pos_students=int(g.y.sum()),Prevalence=g.y.mean(),**capacity(g.y,g.probability,g.id_student.map(lambda x:f'{x:012d}'))))
    full=pd.concat(allresults,ignore_index=True);sel=pd.concat(selected,ignore_index=True)
    assert len(full)==40 and full.completed_new_trials.sum()==900
    cohort=pd.read_csv(out/'cohort_summary.csv');sel=sel.merge(cohort,on=['landmark','task'],validate='one_to_one')
    full.to_csv(out/'all_models_all_horizons.csv',index=False);sel.to_csv(out/'selected_models_trajectory.csv',index=False)
    sdf=pd.DataFrame(stable_results);sdf.to_csv(out/'stable_all_models_trajectory.csv',index=False)
    sdf.loc[sdf.selected_on_validation].to_csv(out/'stable_selected_trajectory.csv',index=False)
    pd.DataFrame(student_top).to_csv(out/'withdrawal_student_level_capacity.csv',index=False)
    for task in TASKS:sel.loc[sel.task.eq(task)].to_csv(out/f'{task}_trajectory.csv',index=False)
    gains=[]
    for kind,df in [('dynamic',sel),('stable',sdf.loc[sdf.selected_on_validation])]:
        for task in TASKS:
            series=df.loc[df.task.eq(task)].set_index('landmark')
            for a,b in zip(HORIZONS[:-1],HORIZONS[1:]):
                row=dict(cohort_type=kind,task=task,from_day=a,to_day=b,from_model=series.loc[a,'model'],to_model=series.loc[b,'model'])
                for metric in ['AP','nAP','AUROC','Brier','Recall_at_10']:row['delta_'+metric]=series.loc[b,metric]-series.loc[a,metric]
                gains.append(row)
    pd.DataFrame(gains).to_csv(out/'adjacent_horizon_gains.csv',index=False)
    stable_deltas(out)
    frozen_check(out)
    dump(out/'execution_checks.json',{'new_trials':900,'new_models':30,'Day28_retrained':False,'total_result_rows':40,'all_frozen_source_hashes_unchanged':True,'status':'passed'})
    print('E3_SUMMARY_COMPLETE',flush=True)

def stable_deltas(out):
    rows=[]
    for task in TASKS:
        for a,b in zip(HORIZONS[:-1],HORIZONS[1:]):
            fa=pd.read_csv(out/f'day{a}'/f'{task}_stable_test_predictions.csv').sort_values('enrolment_key')
            fb=pd.read_csv(out/f'day{b}'/f'{task}_stable_test_predictions.csv').sort_values('enrolment_key')
            assert list(fa.enrolment_key)==list(fb.enrolment_key)
            assert np.array_equal(fa['y_'+task],fb['y_'+task])
            ma=load(out/f'day{a}'/f'{task}_selection.json')['model'];mb=load(out/f'day{b}'/f'{task}_selection.json')['model']
            y=fa['y_'+task].to_numpy(dtype=int);pa=fa[ma+'_probability'].to_numpy();pb=fb[mb+'_probability'].to_numpy()
            _,inv=np.unique(fa.id_student,return_inverse=True);n=inv.max()+1;rng=np.random.default_rng(SEED);rep=[]
            for _ in range(1000):
                w=np.bincount(rng.integers(0,n,n),minlength=n)[inv];prev=np.average(y,weights=w)
                delta=average_precision_score(y,pb,sample_weight=w)-average_precision_score(y,pa,sample_weight=w)
                rep.append([delta,delta/(1-prev)])
            vals=np.array(rep);lo,hi=np.quantile(vals,[.025,.975],axis=0)
            point=average_precision_score(y,pb)-average_precision_score(y,pa)
            rows.append(dict(task=task,from_day=a,to_day=b,from_model=ma,to_model=mb,delta_AP=point,AP_CI_low=lo[0],AP_CI_high=hi[0],delta_nAP=point/(1-y.mean()),nAP_CI_low=lo[1],nAP_CI_high=hi[1],bootstrap_repeats=1000))
            pd.DataFrame(vals,columns=['delta_AP','delta_nAP']).to_csv(out/f'stable_bootstrap_{task}_{a}_{b}.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'stable_paired_horizon_gains.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','train','summarize'],required=True);ap.add_argument('--oulad');ap.add_argument('--baseline',required=True);ap.add_argument('--output',default=str(Path(__file__).parent));a=ap.parse_args()
    out=Path(a.output).resolve();base=Path(a.baseline).resolve();out.mkdir(exist_ok=True,parents=True)
    if a.stage=='prepare':prepare(a.oulad,base,out)
    elif a.stage=='train':train_new(base,out)
    else:summarize(base,out)

if __name__=='__main__':main()
