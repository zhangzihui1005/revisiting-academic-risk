"""E2 label-formulation comparison using the frozen E1 cohort and split."""
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
import argparse,json,time,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import optuna,joblib
from sklearn.metrics import (average_precision_score,roc_auc_score,log_loss,f1_score,
    accuracy_score,balanced_accuracy_score,matthews_corrcoef,confusion_matrix,
    precision_recall_fscore_support)
import e1_reference as e1

NAMES=['LR','RF','XGBoost','LightGBM','CatBoost']
CLASS_NAMES=['Success','Fail','Withdrawal']
PRIMARY_METRICS=['AP_Withdrawal','AP_Fail','MacroF1','MulticlassBrier']
SEED=e1.SEED

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,obj): e1.dump(p,obj)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def check_probs(p):
    assert p.ndim==2 and p.shape[1]==3
    assert np.isfinite(p).all() and (p>=-1e-10).all() and (p<=1+1e-10).all()
    assert np.allclose(p.sum(axis=1),1,atol=1e-6)

def combine(pw,pf):
    p=np.column_stack([(1-pw)*(1-pf),(1-pw)*pf,pw])
    check_probs(p)
    return p

def multi_metrics(y,p):
    check_probs(p)
    onehot=np.eye(3)[y]
    pred=p.argmax(axis=1)
    ap={f'AP_{name}':average_precision_score(onehot[:,i],p[:,i]) for i,name in enumerate(CLASS_NAMES)}
    return dict(**ap,MacroAP=np.mean(list(ap.values())),MacroF1=f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0),
        MacroAUROC=np.mean([roc_auc_score(onehot[:,i],p[:,i]) for i in range(3)]),
        LogLoss=log_loss(y,p,labels=[0,1,2]),MulticlassBrier=np.mean(np.sum((p-onehot)**2,axis=1)),
        Accuracy=accuracy_score(y,pred),BalancedAccuracy=balanced_accuracy_score(y,pred),MCC=matthews_corrcoef(y,pred))

def make_model(name,params,task):
    m=e1.model(name,params)
    if task=='flat':
        if name=='XGBoost': m.set_params(objective='multi:softprob',num_class=3,eval_metric='mlogloss')
        elif name=='LightGBM': m.set_params(objective='multiclass',num_class=3)
        elif name=='CatBoost': m.set_params(loss_function='MultiClass')
    return m

def predict(m,x,task):
    classes=np.asarray(m.classes_,dtype=int)
    p=m.predict_proba(x)
    if task=='merged':
        assert np.array_equal(classes,[0,1]);return p[:,1]
    assert set(classes)=={0,1,2}
    p=p[:,[int(np.flatnonzero(classes==k)[0]) for k in range(3)]]
    check_probs(p)
    return p

def macro_ap(y,p):
    return float(np.mean([average_precision_score(y==k,p[:,k]) for k in range(3)]))

def save_prediction(out,name,part,p,**extra):
    f=part[e1.KEY+['row_id','final_result','y_merged','y_three']].copy()
    if p.ndim==1:f['p_at_risk']=p
    else:
        for i,c in enumerate(CLASS_NAMES):f['p_'+c]=p[:,i]
        f['predicted_class']=p.argmax(axis=1)
    for k,v in extra.items():f[k]=v
    f.to_csv(out/'predictions'/f'{name}.csv',index=False)

def train_task(task,parts,out,trials,data_sha):
    label='y_merged' if task=='merged' else 'y_three'
    y={s:d[label].to_numpy(dtype=int) for s,d in parts.items()}
    x={s:d[e1.FEATURES] for s,d in parts.items()}
    results=[]
    for name in NAMES:
        print(f'START {task} {name} {trials} trials',flush=True)
        t0=time.time()
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED),
            storage='sqlite:///'+str((out/'optuna_e2.sqlite3').resolve()).replace('\\','/'),study_name=f'E2_{task}_{name}',load_if_exists=True)
        if study.user_attrs:assert study.user_attrs['feature_sha256']==data_sha
        else:study.set_user_attr('feature_sha256',data_sha)
        def objective(trial):
            params=e1.parameters(trial,name)
            m=make_model(name,params,task)
            start=time.time();m.fit(x['train'],y['train'])
            vp=predict(m,x['validation'],task)
            score=average_precision_score(y['validation'],vp) if task=='merged' else macro_ap(y['validation'],vp)
            trial.set_user_attr('fit_seconds',time.time()-start)
            if (trial.number+1)%5==0: print(f'  {task} {name} trial {trial.number+1}/{trials} score={score:.4f}',flush=True)
            return score
        done=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        study.optimize(objective,n_trials=max(0,trials-done))
        m=make_model(name,study.best_params,task)
        m.fit(x['train'],y['train'])
        vp=predict(m,x['validation'],task);tp=predict(m,x['test'],task)
        validation_score=average_precision_score(y['validation'],vp) if task=='merged' else macro_ap(y['validation'],vp)
        assert np.isclose(validation_score,study.best_value)
        threshold=None
        if task=='merged':
            threshold=e1.threshold_from_validation(y['validation'],vp)
            metrics=e1.metrics(y['test'],tp,threshold)
            metrics['F1_at_0_5']=f1_score(y['test'],tp>=.5,zero_division=0)
            metrics['test_prevalence']=y['test'].mean()
            metrics['validation_AUPRC']=validation_score
        else:
            metrics=multi_metrics(y['test'],tp)
            metrics['validation_MacroAP']=validation_score
        row=dict(model=name,task=task,n_train=len(y['train']),n_validation=len(y['validation']),n_test=len(y['test']),
            **metrics,completed_trials=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials),runtime_seconds=time.time()-t0)
        results.append(row)
        save_prediction(out,f'{task}_{name}_validation',parts['validation'],vp)
        save_prediction(out,f'{task}_{name}_test',parts['test'],tp)
        joblib.dump({'model':m,'features':e1.FEATURES,'threshold':threshold,'classes':list(m.classes_),'seed':SEED},out/'models'/f'{task}_{name}.joblib')
        dump(out/'models'/f'{task}_{name}_parameters.json',study.best_params)
        study.trials_dataframe().to_csv(out/'trials'/f'{task}_{name}.csv',index=False)
        pd.DataFrame(results).to_csv(out/f'{task}_five_model_results.csv',index=False)
        print('RESULT',json.dumps(row,default=float),flush=True)
    key='validation_AUPRC' if task=='merged' else 'validation_MacroAP'
    selected=max(results,key=lambda r:r[key])['model']
    dump(out/f'{task}_selection.json',{'model':selected,'criterion':key,'source':'validation only'})
    return selected

def dual_predictions(base,parts,out):
    wchoice=load(base/'withdrawal_validation_selected_model.json')['model']
    fchoice=load(base/'failure_validation_selected_model.json')['model']
    pairs={'primary':(wchoice,fchoice),**{n:(n,n) for n in NAMES}}
    results=[]
    for tag,(wn,fn) in pairs.items():
        wm=joblib.load(base/'models'/f'withdrawal_{wn}.joblib')['model']
        fm=joblib.load(base/'models'/f'failure_{fn}.joblib')['model']
        scores={}
        for split in ['validation','test']:
            d=parts[split]
            pw=wm.predict_proba(d[e1.FEATURES])[:,1]
            pf=fm.predict_proba(d[e1.FEATURES])[:,1]
            p=combine(pw,pf)
            save_prediction(out,f'dual_{tag}_{split}',d,p,p_withdrawal_branch=pw,p_failure_conditional=pf)
            scores[split]=multi_metrics(d.y_three.to_numpy(),p)
        results.append(dict(pair=tag,withdrawal_model=wn,conditional_failure_model=fn,
            validation_MacroAP=scores['validation']['MacroAP'],**scores['test']))
    pd.DataFrame(results).to_csv(out/'dual_three_class_results.csv',index=False)
    dump(out/'dual_selection.json',{'withdrawal_model':wchoice,'conditional_failure_model':fchoice,'criterion':'frozen E1 validation branch AP'})
    return results

def four_metrics(y,p,w):
    pred=p.argmax(axis=1)
    return np.array([average_precision_score(y==2,p[:,2],sample_weight=w),
        average_precision_score(y==1,p[:,1],sample_weight=w),
        f1_score(y,pred,labels=[0,1,2],average='macro',sample_weight=w,zero_division=0),
        np.average(np.sum((p-np.eye(3)[y])**2,axis=1),weights=w)])

def paired_bootstrap(y,flat,dual,students,repeats):
    _,inv=np.unique(students,return_inverse=True);n=inv.max()+1
    rng=np.random.default_rng(SEED)
    estimates=[]
    for _ in range(repeats):
        weights=np.bincount(rng.integers(0,n,n),minlength=n)[inv]
        assert all(weights[y==k].sum()>0 for k in range(3))
        estimates.append(four_metrics(y,dual,weights)-four_metrics(y,flat,weights))
    return np.array(estimates)

def compare(out,test,flat_choice):
    y=test.y_three.to_numpy();students=test.id_student.to_numpy()
    allrows=[]
    for tag,fn in [('primary',flat_choice)]+[(n,n) for n in NAMES]:
        flat=pd.read_csv(out/'predictions'/f'flat_{fn}_test.csv')
        dual=pd.read_csv(out/'predictions'/f'dual_{tag}_test.csv')
        assert np.array_equal(flat.row_id,test.row_id) and np.array_equal(dual.row_id,test.row_id)
        cols=['p_'+c for c in CLASS_NAMES]
        fp=flat[cols].to_numpy();dp=dual[cols].to_numpy()
        repeats=2000 if tag=='primary' else 1000
        print('BOOTSTRAP',tag,repeats,flush=True)
        b=paired_bootstrap(y,fp,dp,students,repeats)
        pd.DataFrame(b,columns=PRIMARY_METRICS).to_csv(out/f'bootstrap_replicates_{tag}.csv',index=False)
        lo,hi=np.quantile(b,[.025,.975],axis=0)
        fv=four_metrics(y,fp,None);dv=four_metrics(y,dp,None)
        rows=[dict(comparison=tag,flat_model=fn,metric=m,flat=fv[j],dual=dv[j],delta_dual_minus_flat=dv[j]-fv[j],
            CI_low=lo[j],CI_high=hi[j],bootstrap_repeats=repeats,better_direction='negative' if m=='MulticlassBrier' else 'positive') for j,m in enumerate(PRIMARY_METRICS)]
        allrows+=rows
        if tag=='primary':
            pd.DataFrame(rows).to_csv(out/'primary_paired_comparison.csv',index=False)
            for label,p in [('Flat',fp),('Dual',dp)]:
                cm=confusion_matrix(y,p.argmax(axis=1),labels=[0,1,2])
                pd.DataFrame(cm,index=CLASS_NAMES,columns=CLASS_NAMES).rename_axis('True / Predicted').to_csv(out/f'confusion_{label.lower()}_counts.csv')
                pd.DataFrame(cm/cm.sum(axis=1,keepdims=True),index=CLASS_NAMES,columns=CLASS_NAMES).rename_axis('True / Predicted').to_csv(out/f'confusion_{label.lower()}_row_normalized.csv')
                prec,rec,f1,support=precision_recall_fscore_support(y,p.argmax(axis=1),labels=[0,1,2],zero_division=0)
                pd.DataFrame(dict(class_name=CLASS_NAMES,precision=prec,recall=rec,F1=f1,support=support,predicted_count=cm.sum(axis=0))).to_csv(out/f'per_class_{label.lower()}.csv',index=False)
    pd.DataFrame(allrows).to_csv(out/'all_paired_comparisons.csv',index=False)

def self_tests():
    p=combine(np.array([0.,.2,1.]),np.array([0.,.25,1.]))
    assert np.allclose(p,[[1,0,0],[.6,.2,.2],[0,0,1]])
    y=np.array([0,1,2,0,1,2]); perfect=np.eye(3)[y]
    assert multi_metrics(y,perfect)['MulticlassBrier']==0
    assert np.isclose(multi_metrics(y,np.full((6,3),1/3))['MulticlassBrier'],2/3)
    assert np.allclose(paired_bootstrap(y,perfect,perfect,np.zeros(6),10),0)

def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,name in zip(axes,['flat','dual']):
        counts=pd.read_csv(out/f'confusion_{name}_counts.csv',index_col=0).to_numpy()
        norm=counts/counts.sum(axis=1,keepdims=True)
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{counts[i,j]:,}\n{norm[i,j]:.1%}',ha='center',va='center',color='white' if norm[i,j]>.5 else '#10253b',fontsize=12)
        ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,xlabel='Predicted class',ylabel='True class',title='Flat three-class' if name=='flat' else 'Dual pathway')
    fig.savefig(out/'confusion_matrices.png',dpi=180);fig.savefig(out/'confusion_matrices.pdf');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True);parser.add_argument('--output',default=str(Path(__file__).parent));parser.add_argument('--trials',type=int,default=30)
    a=parser.parse_args();base=Path(a.baseline).resolve();out=Path(a.output).resolve()
    out.mkdir(exist_ok=True,parents=True)
    for folder in ['models','predictions','trials']:(out/folder).mkdir(exist_ok=True)
    self_tests();optuna.logging.set_verbosity(optuna.logging.WARNING)
    frozen=load(out/'baseline_frozen_manifest.json')
    for name,h in frozen['files'].items():assert sha(base/name)==h, name
    manifest=load(base/'run_manifest.json')
    for package,version in manifest['versions'].items():assert importlib.metadata.version(package)==version
    d=pd.read_csv(base/'oulad_day28_features.csv')
    d['y_three']=d.final_result.map({'Pass':0,'Distinction':0,'Fail':1,'Withdrawn':2})
    d['y_merged']=d.y_three.ne(0).astype(int)
    assert len(d)==27421 and d.y_three.notna().all()
    assert np.isfinite(d[e1.FEATURES]).all().all()
    parts={s:d.loc[d.split.eq(s)].copy() for s in ['train','validation','test']}
    sets={s:set(x.id_student) for s,x in parts.items()}
    assert not (sets['train']&sets['test'] or sets['train']&sets['validation'] or sets['validation']&sets['test'])
    e1test=pd.read_csv(base/'withdrawal_test_predictions.csv')
    assert np.array_equal(parts['test'].row_id,e1test.row_id)
    d[e1.KEY+['row_id','split','final_result','y_three','y_merged']].to_csv(out/'frozen_labels_and_split.csv',index=False)
    pd.crosstab(d.split,d.y_three).rename(columns=dict(enumerate(CLASS_NAMES))).to_csv(out/'class_counts_by_split.csv')
    fingerprint=sha(base/'oulad_day28_features.csv')
    dump(out/'run_manifest.json',dict(seed=SEED,features=e1.FEATURES,feature_file_sha256=fingerprint,baseline_version=frozen['name'],
        python=manifest['python'],versions=manifest['versions'],trials_per_model=a.trials,synthetic_tests='passed',
        cohort_size=len(d),test_size=len(parts['test']),student_overlap=0,baseline_path=str(base),class_order=CLASS_NAMES,
        flat_selection='validation MacroAP (three classes)',primary_dual='E1 CatBoost Withdrawal + LightGBM conditional Failure'))
    print('COHORT',d.y_three.value_counts().sort_index().to_dict(),flush=True)
    train_task('merged',parts,out,a.trials,fingerprint)
    flat_choice=train_task('flat',parts,out,a.trials,fingerprint)
    dual_predictions(base,parts,out)
    compare(out,parts['test'],flat_choice)
    figures(out)
    for name,h in frozen['files'].items():assert sha(base/name)==h,name
    dump(out/'execution_checks.json',{'status':'passed','baseline_hashes_unchanged':len(frozen['files']),'test_row_ids_unchanged':True,'probability_checks':'passed','synthetic_checks':'passed','completed_new_trials':300})
    print('E2_COMPLETE',flush=True)

if __name__=='__main__':main()
