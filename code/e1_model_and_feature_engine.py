"""Reproducible E1. Run with Python 3.12 and requirements-lock.txt.
python run_experiment.py --oulad OULAD.zip --uci UCI.zip --output results
No test-set tuning. Only Day28 E1 is run; later experiments are deliberately separate.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '6')
import argparse, hashlib, json, time, zipfile, platform, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import joblib
from scipy.stats import entropy
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, brier_score_loss,
    precision_recall_curve, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 20261007
H = 28
KEY = ['code_module', 'code_presentation', 'id_student']
FEATURES = ['total_clicks','active_days','active_ratio','unique_sites',
    'unique_activity_types','clicks_per_active_day','daily_click_std','inactivity_gap',
    'recent7_clicks','previous7_clicks','engagement_change','engagement_slope',
    'activity_entropy','num_of_prev_attempts','studied_credits','registration_lead_time']

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x,'item') else str(x)), encoding='utf-8')

def sha(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(z, name):
    return pd.read_csv(z.open(name), na_values=['?'])

def active_cohort(info, reg, cutoff=H):
    df = info.merge(reg, on=KEY, validate='one_to_one')
    reasons = np.select([
        df.date_registration.isna(),
        df.date_registration.gt(cutoff),
        df.date_unregistration.le(cutoff),
        df.final_result.eq('Withdrawn') & df.date_unregistration.isna(),
        df.date_unregistration.notna() & df.final_result.ne('Withdrawn')],
        ['unknown_registration','registered_after_cutoff','withdrawn_by_cutoff',
         'withdrawal_date_unknown','unregistration_label_inconsistent'], default='included')
    df['cohort_status'] = reasons
    included = df.loc[df.cohort_status.eq('included')].copy().reset_index(drop=True)
    included['row_id'] = np.arange(len(included))
    included['y_withdrawal'] = included.final_result.eq('Withdrawn').astype(int)
    included['y_failure'] = np.where(included.final_result.eq('Withdrawn'), np.nan, included.final_result.eq('Fail').astype(int))
    return included, df

def features_from_logs(cohort, logs, sites, cutoff=H):
    # Apply cutoff BEFORE any log-derived aggregation. Day0..Day28 inclusive.
    logs = logs.loc[logs.date.between(0, cutoff)].copy()
    logs = logs.merge(cohort[KEY+['row_id','date_registration']], on=KEY, validate='many_to_one')
    logs = logs.loc[logs.date.ge(logs.date_registration)].copy()
    logs = logs.merge(sites[['id_site','code_module','code_presentation','activity_type']],
        on=['id_site','code_module','code_presentation'], how='left', validate='many_to_one')
    assert logs.activity_type.notna().all(), 'Unmatched resource metadata'
    assert logs.date.between(0,cutoff).all()
    n = len(cohort)
    daily = np.zeros((n, cutoff+1), dtype=float)
    daily_agg = logs.groupby(['row_id','date']).sum_click.sum()
    if len(daily_agg):
        daily[daily_agg.index.get_level_values(0),daily_agg.index.get_level_values(1)] = daily_agg.values
    start = np.maximum(0,cohort.date_registration.to_numpy(dtype=int))
    days = np.arange(cutoff+1)
    exposure = days[None,:] >= start[:,None]
    duration = exposure.sum(axis=1)
    total = daily.sum(axis=1)
    active = (daily>0).sum(axis=1)
    out = cohort[KEY+['row_id','final_result','y_withdrawal','y_failure']].copy()
    out['total_clicks'] = total
    out['active_days'] = active
    out['active_ratio'] = active/duration
    out['unique_sites'] = out.row_id.map(logs.groupby('row_id').id_site.nunique()).fillna(0)
    out['unique_activity_types'] = out.row_id.map(logs.groupby('row_id').activity_type.nunique()).fillna(0)
    out['clicks_per_active_day'] = np.divide(total,active,out=np.zeros(n),where=active>0)
    out['daily_click_std'] = np.sqrt(np.maximum(0,(daily*daily).sum(axis=1)/duration-(total/duration)**2))
    last_day = np.where(daily>0,days[None,:],-1).max(axis=1)
    out['inactivity_gap'] = np.where(active>0,cutoff-last_day,duration)
    out['recent7_clicks'] = daily[:,max(0,cutoff-6):cutoff+1].sum(axis=1)
    out['previous7_clicks'] = daily[:,max(0,cutoff-13):max(0,cutoff-6)].sum(axis=1)
    out['engagement_change'] = np.log1p(out.recent7_clicks)-np.log1p(out.previous7_clicks)
    mean_x = (days[None,:]*exposure).sum(axis=1)/duration
    centered = (days[None,:]-mean_x[:,None])*exposure
    denom = (centered**2).sum(axis=1)
    out['engagement_slope'] = np.divide((centered*daily).sum(axis=1),denom,out=np.zeros(n),where=denom>0)
    activities = logs.groupby(['row_id','activity_type']).sum_click.sum()
    ent = activities.groupby(level=0).agg(lambda x: entropy(x.to_numpy()))
    out['activity_entropy'] = out.row_id.map(ent).fillna(0)
    out['num_of_prev_attempts'] = cohort.num_of_prev_attempts
    out['studied_credits'] = cohort.studied_credits
    out['registration_lead_time'] = -cohort.date_registration
    assert np.isfinite(out[FEATURES].to_numpy()).all()
    assert not out.duplicated(KEY).any()
    audit = {'retained_log_rows':len(logs),'retained_log_min_day':int(logs.date.min()) if len(logs) else None,
        'retained_log_max_day':int(logs.date.max()) if len(logs) else None,'zero_activity_records':int((active==0).sum())}
    return out,audit

def test_temporal_features():
    # Future rows must not affect ANY feature; zeros must contribute to temporal statistics.
    c = pd.DataFrame([dict(code_module='A',code_presentation='P',id_student=1,row_id=0,
        date_registration=-2,final_result='Pass',y_withdrawal=0,y_failure=0,
        num_of_prev_attempts=0,studied_credits=60)])
    v = pd.DataFrame([dict(id_site=1,code_module='A',code_presentation='P',activity_type='resource')])
    logs = pd.DataFrame([dict(code_module='A',code_presentation='P',id_student=1,id_site=1,date=d,sum_click=k)
        for d,k in [(0,2),(28,3),(29,99999),(-1,88888)]])
    a,_ = features_from_logs(c,logs,v)
    b,_ = features_from_logs(c,logs.loc[logs.date.between(0,28)],v)
    pd.testing.assert_frame_equal(a,b)
    assert a.total_clicks.iloc[0]==5 and a.active_days.iloc[0]==2
    assert a.recent7_clicks.iloc[0]==3 and a.previous7_clicks.iloc[0]==0
    assert np.isclose(a.active_ratio.iloc[0],2/29)
    assert np.isclose(a.daily_click_std.iloc[0],np.std([2]+[0]*27+[3]))
    empty,_ = features_from_logs(c,logs.iloc[:0],v)
    assert empty.inactivity_gap.iloc[0]==29 and empty.activity_entropy.iloc[0]==0

def prepare(oulad, out):
    with zipfile.ZipFile(oulad) as z:
        info,reg,sites = read(z,'studentInfo.csv'),read(z,'studentRegistration.csv'),read(z,'vle.csv')
        cohort,all_records = active_cohort(info,reg)
        all_records[KEY+['date_registration','date_unregistration','final_result','cohort_status']].to_csv(out/'cohort_audit.csv',index=False)
        chunks=[]
        for chunk in pd.read_csv(z.open('studentVle.csv'), na_values=['?'], chunksize=500000):
            chunks.append(chunk.loc[chunk.date.between(0,H)])
        data,log_audit = features_from_logs(cohort,pd.concat(chunks,ignore_index=True),sites)
    split = np.empty(len(data),dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    for fold,(_,idx) in enumerate(splitter.split(data,data.final_result,data.id_student)):
        split[idx] = 'test' if fold==0 else 'validation' if fold==1 else 'train'
    data['split'] = split
    sets={s:set(data.loc[data.split.eq(s),'id_student']) for s in ['train','validation','test']}
    assert not (sets['train']&sets['test'] or sets['train']&sets['validation'] or sets['test']&sets['validation'])
    data.to_csv(out/'oulad_day28_features.csv',index=False)
    splits=[]
    for task in ['withdrawal','failure']:
        for s in ['train','validation','test']:
            d=data.loc[data.split.eq(s)&data['y_'+task].notna()]
            splits.append(dict(task=task,split=s,records=len(d),students=d.id_student.nunique(),positive=int(d['y_'+task].sum()),prevalence=float(d['y_'+task].mean())))
    pd.DataFrame(splits).to_csv(out/'split_summary.csv',index=False)
    audit={'raw_records':len(info),'included_records':len(data),'unique_students':data.id_student.nunique(),
        'cohort_counts':all_records.cohort_status.value_counts().to_dict(),
        'included_labels':data.final_result.value_counts().to_dict(),
        'student_overlap_train_validation_test':0,'feature_count':len(FEATURES),**log_audit}
    dump(out/'data_audit.json',audit)
    print('DATA_AUDIT',json.dumps(audit,default=int),flush=True)
    return data

def model(name,p):
    if name=='LR':
        return make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,solver='lbfgs',random_state=SEED,**p))
    if name=='RF': return RandomForestClassifier(n_jobs=6,random_state=SEED,**p)
    if name=='XGBoost': return XGBClassifier(n_jobs=6,random_state=SEED,tree_method='hist',eval_metric='logloss',**p)
    if name=='LightGBM': return LGBMClassifier(n_jobs=6,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True,subsample_freq=1,**p)
    if name=='CatBoost': return CatBoostClassifier(thread_count=6,random_seed=SEED,verbose=False,allow_writing_files=False,loss_function='Logloss',**p)
    raise ValueError(name)

def parameters(trial,name):
    if name=='LR': return dict(C=trial.suggest_float('C',1e-4,100,log=True),class_weight=trial.suggest_categorical('class_weight',[None,'balanced']))
    if name=='RF': return dict(n_estimators=trial.suggest_int('n_estimators',150,400,step=50),max_depth=trial.suggest_int('max_depth',5,24),min_samples_leaf=trial.suggest_int('min_samples_leaf',2,30),max_features=trial.suggest_categorical('max_features',['sqrt',0.7,1.0]),class_weight=trial.suggest_categorical('class_weight',[None,'balanced']))
    common=dict(learning_rate=trial.suggest_float('learning_rate',0.02,0.2,log=True))
    if name=='CatBoost': return dict(**common,iterations=trial.suggest_int('iterations',150,500,step=50),depth=trial.suggest_int('depth',4,7),l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1,20,log=True),auto_class_weights=trial.suggest_categorical('auto_class_weights',['None','Balanced']))
    common.update(n_estimators=trial.suggest_int('n_estimators',150,500,step=50),subsample=trial.suggest_float('subsample',0.7,1),colsample_bytree=trial.suggest_float('colsample_bytree',0.7,1),reg_lambda=trial.suggest_float('reg_lambda',0.1,20,log=True))
    if name=='XGBoost': return dict(**common,max_depth=trial.suggest_int('max_depth',3,7),min_child_weight=trial.suggest_float('min_child_weight',1,20,log=True))
    return dict(**common,num_leaves=trial.suggest_int('num_leaves',15,63),min_child_samples=trial.suggest_int('min_child_samples',20,100))

def threshold_from_validation(y,p):
    precision,recall,t = precision_recall_curve(y,p)
    f = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    return float(t[np.argmax(f)])

def metrics(y,p,threshold):
    pred=p>=threshold
    return dict(AUPRC=average_precision_score(y,p),AUROC=roc_auc_score(y,p),
        Recall=recall_score(y,pred,zero_division=0),Precision=precision_score(y,pred,zero_division=0),
        F1=f1_score(y,pred,zero_division=0),MCC=matthews_corrcoef(y,pred),
        BalancedAccuracy=balanced_accuracy_score(y,pred),Brier=brier_score_loss(y,p),threshold=threshold)

def bootstrap_ap(y,p,students,repeats=1000):
    # Cluster bootstrap students; all enrolments of a sampled student travel together.
    _,inv=np.unique(students,return_inverse=True)
    count=inv.max()+1
    rng=np.random.default_rng(SEED)
    vals=[]
    for _ in range(repeats):
        w=np.bincount(rng.integers(0,count,size=count),minlength=count)[inv]
        if np.sum(w*y)>0 and np.sum(w*(1-y))>0:
            vals.append(average_precision_score(y,p,sample_weight=w))
    return np.quantile(vals,[0.025,0.975]).tolist()

def train_all(data,out,trials):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for task in ['withdrawal','failure']:
        d=data.loc[data['y_'+task].notna()].copy()
        parts={s:d.loc[d.split.eq(s)] for s in ['train','validation','test']}
        X={s:f[FEATURES] for s,f in parts.items()}
        y={s:f['y_'+task].astype(int).to_numpy() for s,f in parts.items()}
        results=[]
        predictions=parts['test'][KEY+['row_id','final_result','y_'+task]].copy()
        for name in ['LR','RF','XGBoost','LightGBM','CatBoost']:
            print(f'START {task} {name}: {trials} trials',flush=True)
            start=time.time()
            study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED),
                storage='sqlite:///'+str((out/'optuna_studies.sqlite3').resolve()).replace('\\','/'),
                study_name=f'{task}_{name}',load_if_exists=True)
            def objective(trial):
                p=parameters(trial,name)
                m=model(name,p)
                t0=time.time()
                m.fit(X['train'],y['train'])
                score=average_precision_score(y['validation'],m.predict_proba(X['validation'])[:,1])
                trial.set_user_attr('fit_seconds',time.time()-t0)
                if (trial.number+1)%5==0: print(f'  {task} {name} trial {trial.number+1}/{trials} AP={score:.4f}',flush=True)
                return score
            remaining=max(0,trials-len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]))
            study.optimize(objective,n_trials=remaining)
            best=model(name,study.best_params)
            best.fit(X['train'],y['train'])
            vp=best.predict_proba(X['validation'])[:,1]
            threshold=threshold_from_validation(y['validation'],vp)
            tp=best.predict_proba(X['test'])[:,1]
            assert np.isfinite(tp).all() and ((tp>=0)&(tp<=1)).all()
            lo,hi=bootstrap_ap(y['test'],tp,parts['test'].id_student.to_numpy())
            result=dict(task=task,model=name,n_train=len(y['train']),n_validation=len(y['validation']),n_test=len(y['test']),
                test_prevalence=float(y['test'].mean()),validation_AUPRC=study.best_value,
                **metrics(y['test'],tp,threshold),AUPRC_CI_low=lo,AUPRC_CI_high=hi,
                F1_at_0_5=f1_score(y['test'],tp>=0.5,zero_division=0),
                completed_trials=len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]),
                runtime_seconds=time.time()-start)
            results.append(result)
            predictions[name+'_probability']=tp
            pd.DataFrame(results).to_csv(out/f'oulad_day28_{task}_results.csv',index=False)
            predictions.to_csv(out/f'{task}_test_predictions.csv',index=False)
            pd.DataFrame(study.trials_dataframe()).to_csv(out/f'trials_{task}_{name}.csv',index=False)
            joblib.dump({'model':best,'features':FEATURES,'threshold':threshold,'seed':SEED},out/'models'/f'{task}_{name}.joblib')
            dump(out/'models'/f'{task}_{name}_parameters.json',study.best_params)
            print('RESULT',json.dumps(result),flush=True)
        chosen=max(results,key=lambda r:r['validation_AUPRC'])['model']
        dump(out/f'{task}_validation_selected_model.json',{'model':chosen,'selection':'validation AUPRC only'})

def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for i,task in enumerate(['withdrawal','failure']):
        d=pd.read_csv(out/f'{task}_test_predictions.csv')
        y=d['y_'+task]
        for name in ['LR','RF','XGBoost','LightGBM','CatBoost']:
            p=d[name+'_probability']
            prec,rec,_=precision_recall_curve(y,p)
            axes[i,0].plot(rec,prec,label=f'{name} AP={average_precision_score(y,p):.3f}',linewidth=1.5)
            fpr,tpr,_=roc_curve(y,p)
            axes[i,1].plot(fpr,tpr,label=f'{name} AUC={roc_auc_score(y,p):.3f}',linewidth=1.5)
        axes[i,0].axhline(y.mean(),color='gray',linestyle='--',label=f'Prevalence {y.mean():.3f}')
        axes[i,1].plot([0,1],[0,1],color='gray',linestyle='--')
        axes[i,0].set(xlabel='Recall',ylabel='Precision',title=task.title()+' — Day28 precision-recall')
        axes[i,1].set(xlabel='False positive rate',ylabel='True positive rate',title=task.title()+' — Day28 ROC')
        for ax in axes[i]:
            ax.legend(fontsize=8); ax.grid(alpha=.2); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.savefig(out/'day28_test_curves.png',dpi=180)
    fig.savefig(out/'day28_test_curves.pdf')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--oulad',required=True)
    ap.add_argument('--uci',required=True)
    ap.add_argument('--output',default=str(Path(__file__).parent))
    ap.add_argument('--trials',type=int,default=30)
    args=ap.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    test_temporal_features()
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','optuna','xgboost','lightgbm','catboost','matplotlib','joblib']}
    dump(out/'run_manifest.json',dict(seed=SEED,horizon=H,features=FEATURES,trials=args.trials,python=platform.python_version(),versions=versions,
        inputs={str(p):sha(p) for p in [args.oulad,args.uci]},temporal_unit_tests='passed'))
    from check_uci import check_uci
    check_uci(args.uci,out)
    data=prepare(args.oulad,out)
    train_all(data,out,args.trials)
    figures(out)
    print('EXPERIMENT_1_COMPLETE',flush=True)

if __name__=='__main__': main()
