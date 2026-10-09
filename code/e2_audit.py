"""Independent consistency checks for E2 saved models, probabilities and paired intervals."""
import sys
sys.dont_write_bytecode=True
import argparse,json,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss,f1_score,confusion_matrix,precision_recall_curve

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
NAMES=['LR','RF','XGBoost','LightGBM','CatBoost']
CLASSES=['Success','Fail','Withdrawal']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--output',default=str(Path(__file__).parent));a=ap.parse_args()
    base=Path(a.baseline);out=Path(a.output)
    manifest=load(out/'run_manifest.json');features=manifest['features']
    frozen=load(out/'baseline_frozen_manifest.json')
    for f,h in frozen['files'].items():assert sha(base/f)==h,f
    assert sha(base/'oulad_day28_features.csv')==manifest['feature_file_sha256']
    data=pd.read_csv(base/'oulad_day28_features.csv')
    split=pd.read_csv(out/'frozen_labels_and_split.csv')
    assert np.array_equal(data.row_id,split.row_id) and np.array_equal(data.split,split.split)
    assert np.array_equal(data.final_result,split.final_result)
    parts={s:data.loc[data.split.eq(s)] for s in ['validation','test']}
    mapping={'Pass':0,'Distinction':0,'Fail':1,'Withdrawn':2}
    successes=0
    for task in ['merged','flat']:
        results=pd.read_csv(out/f'{task}_five_model_results.csv')
        assert len(results)==5 and results.completed_trials.eq(30).all()
        for _,r in results.iterrows():
            saved=joblib.load(out/'models'/f'{task}_{r.model}.joblib')
            for s,d in parts.items():
                pred=pd.read_csv(out/'predictions'/f'{task}_{r.model}_{s}.csv')
                assert np.array_equal(pred.row_id,d.row_id)
                y=d.final_result.map(mapping).to_numpy()
                p=saved['model'].predict_proba(d[features])
                if task=='merged':
                    y=(y!=0).astype(int);p=p[:,1]
                    assert np.allclose(p,pred.p_at_risk)
                    apv=average_precision_score(y,p)
                    if s=='validation':
                        assert np.isclose(apv,r.validation_AUPRC)
                        pr,re,t=precision_recall_curve(y,p)
                        f=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-15)
                        assert np.isclose(t[np.argmax(f)],r.threshold)
                    else:
                        assert np.isclose(apv,r.AUPRC)
                        assert np.isclose(roc_auc_score(y,p),r.AUROC)
                else:
                    assert np.array_equal(saved['model'].classes_,[0,1,2])
                    assert np.allclose(p,pred[['p_'+c for c in CLASSES]])
                    assert np.allclose(p.sum(axis=1),1,atol=1e-6)
                    aps=[average_precision_score(y==i,p[:,i]) for i in range(3)]
                    if s=='validation':assert np.isclose(np.mean(aps),r.validation_MacroAP)
                    else:
                        for c,av in zip(CLASSES,aps):assert np.isclose(av,r['AP_'+c])
                        assert np.isclose(f1_score(y,p.argmax(axis=1),average='macro'),r.MacroF1)
                        assert np.isclose(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)),r.MulticlassBrier)
                        assert np.isclose(log_loss(y,p),r.LogLoss)
            successes+=1
        selection=load(out/f'{task}_selection.json')['model']
        key='validation_AUPRC' if task=='merged' else 'validation_MacroAP'
        assert selection==results.loc[results[key].idxmax(),'model']
    dualres=pd.read_csv(out/'dual_three_class_results.csv')
    for _,r in dualres.iterrows():
        wm=joblib.load(base/'models'/f'withdrawal_{r.withdrawal_model}.joblib')['model']
        fm=joblib.load(base/'models'/f'failure_{r.conditional_failure_model}.joblib')['model']
        for s,d in parts.items():
            pred=pd.read_csv(out/'predictions'/f'dual_{r.pair}_{s}.csv')
            assert np.array_equal(pred.row_id,d.row_id)
            pw=wm.predict_proba(d[features])[:,1];pf=fm.predict_proba(d[features])[:,1]
            assert np.allclose(pred.p_withdrawal_branch,pw) and np.allclose(pred.p_failure_conditional,pf)
            expected=np.column_stack([(1-pw)*(1-pf),(1-pw)*pf,pw])
            actual=pred[['p_'+c for c in CLASSES]].to_numpy()
            assert np.allclose(actual,expected) and np.allclose(actual.sum(axis=1),1)
            if s=='test':
                e1w=pd.read_csv(base/'withdrawal_test_predictions.csv')
                e1f=pd.read_csv(base/'failure_test_predictions.csv')
                assert np.allclose(pw,e1w[r.withdrawal_model+'_probability'])
                mask=d.final_result.ne('Withdrawn').to_numpy()
                assert np.array_equal(d.loc[mask,'row_id'],e1f.row_id)
                assert np.allclose(pf[mask],e1f[r.conditional_failure_model+'_probability'])
                y=d.final_result.map(mapping).to_numpy()
                assert np.isclose(np.mean(np.sum((actual-np.eye(3)[y])**2,axis=1)),r.MulticlassBrier)
                for i,c in enumerate(CLASSES):assert np.isclose(average_precision_score(y==i,actual[:,i]),r['AP_'+c])
    comparison=pd.read_csv(out/'all_paired_comparisons.csv')
    for tag,g in comparison.groupby('comparison'):
        boot=pd.read_csv(out/f'bootstrap_replicates_{tag}.csv')
        for _,r in g.iterrows():
            lo,hi=np.quantile(boot[r.metric],[.025,.975])
            assert np.isclose(lo,r.CI_low) and np.isclose(hi,r.CI_high)
            assert np.isclose(r.dual-r.flat,r.delta_dual_minus_flat)
            assert len(boot)==r.bootstrap_repeats
    selected=load(out/'flat_selection.json')['model']
    # Independently verify paired bootstrap via explicit duplicated rows instead of weights.
    flatp=pd.read_csv(out/'predictions'/f'flat_{selected}_test.csv')
    dualp=pd.read_csv(out/'predictions'/'dual_primary_test.csv')
    y=flatp.y_three.to_numpy();pc=['p_'+c for c in CLASSES]
    fp=flatp[pc].to_numpy();dp=dualp[pc].to_numpy()
    _,inv=np.unique(flatp.id_student,return_inverse=True);n=inv.max()+1
    rng=np.random.default_rng(manifest['seed'])
    replicates=pd.read_csv(out/'bootstrap_replicates_primary.csv')
    for b in range(50):
        weights=np.bincount(rng.integers(0,n,n),minlength=n)[inv]
        idx=np.repeat(np.arange(len(y)),weights);yy=y[idx]
        def m(p):
            pp=p[idx]
            return np.array([average_precision_score(yy==2,pp[:,2]),average_precision_score(yy==1,pp[:,1]),
                f1_score(yy,pp.argmax(axis=1),labels=[0,1,2],average='macro',zero_division=0),
                np.mean(np.sum((pp-np.eye(3)[yy])**2,axis=1))])
        assert np.allclose(m(dp)-m(fp),replicates.iloc[b].to_numpy(),atol=1e-10)
    for form,file in [('flat',f'flat_{selected}_test.csv'),('dual','dual_primary_test.csv')]:
        p=pd.read_csv(out/'predictions'/file)
        cm=confusion_matrix(p.y_three,p.predicted_class,labels=[0,1,2])
        assert np.array_equal(cm,pd.read_csv(out/f'confusion_{form}_counts.csv',index_col=0))
    checks={'status':'passed','E1_files_unchanged':len(frozen['files']),'new_models_reloaded_and_checked':successes,
        'dual_pairs_checked':len(dualres),'test_records_unchanged':len(parts['test']),'all_300_trials_complete':True,
        'probabilities_and_metrics_reproduced':True,'paired_percentile_intervals_verified':True,'paired_replicates_verified_by_explicit_rows':50,'E1_branch_predictions_reproduced':True}
    (out/'independent_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks))

if __name__=='__main__':main()
