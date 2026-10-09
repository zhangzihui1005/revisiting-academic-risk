"""Independent membership and numeric audit, without reusing split/evaluate code."""
import sys
sys.dont_write_bytecode=True
import argparse,json,hashlib,sqlite3,itertools,gc
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,f1_score
ROOT=Path(__file__).resolve().parent
MOD=[s*3 for s in 'ABCDEFG'];TASK=['withdrawal','failure'];NAMES=['LR','RF','XGBoost','LightGBM','CatBoost']
def readj(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):assert np.isclose(a,b,rtol=1e-9,atol=1e-10),(a,b)
def audit(final):
    manifest=readj(ROOT/'run_manifest.json');d=pd.read_csv(ROOT/'day28_features.csv');features=manifest['features']
    assert len(d)==27421 and d.id_student.nunique()==24749 and len(features)==16
    assert not d.duplicated(['code_module','code_presentation','id_student']).any()
    assert np.isfinite(d[features]).all().all()
    assert set(features).isdisjoint({'code_module','code_presentation','id_student','row_id','final_result','y_withdrawal','y_failure','split'})
    assert (d.y_withdrawal==d.final_result.eq('Withdrawn')).all()
    assert d.y_failure.isna().equals(d.final_result.eq('Withdrawn'))
    f=d.loc[d.y_failure.notna()];assert (f.y_failure==f.final_result.eq('Fail')).all()
    for name,key in [('PROTOCOL.txt','protocol_sha256'),('day28_features.csv','feature_sha256'),('e1_reference.py','reference_code_sha256'),('run_e4a.py','training_code_sha256')]:assert sha(ROOT/name)==manifest[key]
    for rel,h in manifest['split_hashes'].items():assert sha(ROOT/rel)==h
    frozen=readj(ROOT/'frozen_sources.json');nfiles=0
    for group in frozen.values():
        for rel,h in group['files'].items():assert sha(Path(group['directory'])/rel)==h;nfiles+=1
    assert manifest['feature_sha256']==frozen['E1']['files']['oulad_day28_features.csv']
    assert manifest['reference_code_sha256']==frozen['E1']['files']['run_experiment.py']
    oa=pd.read_csv(ROOT/'outer_fold_audit.csv');ia=pd.read_csv(ROOT/'inner_fold_audit.csv')
    assert len(oa)==14 and len(ia)==84
    folds=0;inners=0;trials=0;domain_scores=0;models=0
    if final:
        res=pd.read_csv(ROOT/'all_models_test_results.csv');selected=pd.read_csv(ROOT/'inner_selected_models.csv');tc=readj(ROOT/'training_complete.json')
        assert sha(ROOT/'inner_selected_models.csv')==tc['selection_sha256']
        assert len(res)==70 and len(selected)==14
    for g,t in itertools.product(MOD,TASK):
        held=set(d.loc[d.code_module==g,'id_student']);base=ROOT/'folds'/g/t
        assert held==set(pd.read_csv(base.parent/'heldout_students.csv').id_student)
        sr=d.loc[(d.code_module!=g)&~d.id_student.isin(held)]
        tr=sr.loc[sr['y_'+t].notna()];te=d.loc[(d.code_module==g)&d['y_'+t].notna()]
        assert set(tr.id_student).isdisjoint(held)
        for name,df in [('train',tr),('test',te)]:
            saved=pd.read_csv(base/f'{name}_membership.csv')
            assert set(saved.row_id)==set(df.row_id) and len(saved)==len(df)
            assert set(pd.read_csv(base/f'{name}_students.csv').id_student)==set(df.id_student)
        ar=oa.loc[(oa.heldout_module==g)&(oa.task==t)].iloc[0]
        for col,value in [('train_records',len(tr)),('train_students',tr.id_student.nunique()),('test_records',len(te)),('test_students',te.id_student.nunique()),('train_test_student_overlap',0),('test_positive',te['y_'+t].sum()),('test_prevalence',te['y_'+t].mean())]:close(ar[col],value)
        innerframes={}
        for h in MOD:
            if h==g:continue
            vstudents=set(sr.loc[sr.code_module==h,'id_student'])
            it=tr.loc[(tr.code_module!=h)&~tr.id_student.isin(vstudents)];iv=tr.loc[tr.code_module==h]
            assert set(it.id_student).isdisjoint(vstudents)
            assert set(iv.id_student).isdisjoint(held) and set(it.id_student).isdisjoint(held)
            assert it['y_'+t].nunique()==2 and iv['y_'+t].nunique()==2 and te['y_'+t].nunique()==2
            for role,df in [('train',it),('validation',iv)]:
                saved=pd.read_csv(base/f'inner_{h}_{role}_membership.csv')
                assert set(saved.row_id)==set(df.row_id) and len(saved)==len(df)
            ir=ia.loc[(ia.heldout_module==g)&(ia.task==t)&(ia.validation_module==h)].iloc[0]
            for col,value in [('train_records',len(it)),('validation_records',len(iv)),('train_positive',it['y_'+t].sum()),('validation_positive',iv['y_'+t].sum()),('train_validation_student_overlap',0),('inner_train_outer_test_student_overlap',0),('inner_validation_outer_test_student_overlap',0)]:close(ir[col],value)
            innerframes[h]=(it,iv);inners+=1
        folds+=1
        if not final:continue
        candidates=[]
        for name in NAMES:
            dest=base/name;s=readj(dest/'selection.json');candidates.append(s)
            assert s['completed_trials']==15 and sha(dest/'model.joblib')==s['model_sha256']
            with sqlite3.connect(dest/'study.sqlite3') as db:
                count=db.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'").fetchone()[0]
                assert count==15
                trialvals=pd.read_sql_query('SELECT t.number, v.value FROM trials t JOIN trial_values v ON t.trial_id=v.trial_id ORDER BY t.number',db)
            scores=pd.read_csv(dest/'trial_inner_domains.csv');assert len(scores)==90
            means=scores.groupby('trial',sort=True).AP.mean();assert means.idxmax()==s['best_trial']
            for k,z in scores.groupby('trial'):
                close(z.AP.mean(),trialvals.loc[trialvals.number==k,'value'].iloc[0])
                for col,value in [('mean_AP',z.AP.mean()),('SD_AP',z.AP.std(ddof=0)),('Worst_AP',z.AP.min()),('Mean_nAP',z.nAP.mean())]:assert np.allclose(z[col],value)
                assert np.allclose(z.nAP,(z.AP-z.prevalence)/(1-z.prevalence))
            close(means.max(),s['inner_mean_AP']);trials+=15;domain_scores+=90
            oof=pd.read_csv(dest/'best_inner_predictions.csv.gz')
            assert not oof.row_id.duplicated().any() and set(oof.row_id)==set(tr.row_id)
            for h,(_,iv) in innerframes.items():
                o=oof.loc[oof.code_module==h].set_index('row_id').sort_index();v=iv.set_index('row_id').sort_index()
                assert o.index.equals(v.index) and np.array_equal(o.y,v['y_'+t])
                ds=scores.loc[(scores.trial==s['best_trial'])&(scores.validation_module==h)].iloc[0]
                close(average_precision_score(o.y,o.probability),ds.AP)
            # Independently check selected threshold against full saved candidate curve,
            # and explicitly re-evaluate F1 at its optimum and deterministic sampled candidates.
            curve=pd.read_csv(dest/'inner_threshold_curve.csv.gz')
            assert len(curve)==len(np.unique(np.r_[0.,oof.probability,1.]))
            best=curve.mean_domain_F1.idxmax();close(curve.loc[best,'threshold'],s['threshold'])
            exact_candidates=np.unique(np.r_[0.,oof.probability.to_numpy(),1.])
            assert np.allclose(curve.threshold,exact_candidates,rtol=1e-14,atol=1e-15)
            for i in np.unique(np.r_[best,np.linspace(0,len(curve)-1,7,dtype=int)]):
                threshold=exact_candidates[i]
                actual=np.mean([f1_score(z.y,z.probability>=threshold,zero_division=0) for _,z in oof.groupby('code_module')])
                close(actual,curve.loc[i,'mean_domain_F1'])
            obj=joblib.load(dest/'model.joblib');assert obj['features']==features
            if name=='LR':assert np.allclose(obj['model'].named_steps['standardscaler'].mean_,tr[features].mean())
            pred=pd.read_csv(dest/'test_predictions.csv',float_precision='round_trip').set_index('row_id').sort_index();test=te.set_index('row_id').sort_index()
            assert pred.index.equals(test.index) and np.array_equal(pred.y,test['y_'+t])
            pp=obj['model'].predict_proba(test[features])[:,1]
            # pandas serializes float32 using a shortest float32-round-tripping
            # decimal. Restore that dtype before requiring 1e-12 agreement.
            restored=pred.probability.to_numpy().astype(pp.dtype)
            assert np.allclose(pp,restored,rtol=1e-12,atol=1e-12),(g,t,name,float(np.max(abs(pp-restored))))
            r=res.loc[(res.heldout_module==g)&(res.task==t)&(res.model==name)].iloc[0]
            y=pred.y.to_numpy();p=restored;pi=y.mean();ap=average_precision_score(y,p)
            for col,value in [('AP',ap),('nAP',(ap-pi)/(1-pi)),('AUROC',roc_auc_score(y,p)),('Brier',brier_score_loss(y,p)),('F1',f1_score(y,p>=s['threshold'])),('test_N',len(y)),('test_Pos',y.sum()),('test_Prevalence',pi)]:close(r[col],value)
            ordered=pred.sort_values(['probability','enrolment_key'],ascending=[False,True],kind='stable')
            for pct in [5,10,20]:
                k=int(np.ceil(len(pred)*pct/100));hits=ordered.iloc[:k].y.sum()
                for col,value in [(f'Precision_at_{pct}',hits/k),(f'Recall_at_{pct}',hits/y.sum()),(f'Lift_at_{pct}',hits/k/pi),(f'selected_N_at_{pct}',k)]:close(r[col],value)
            models+=1;del obj;gc.collect()
        expected=max(candidates,key=lambda s:s['inner_mean_AP'])['model']
        chosen=selected.loc[(selected.heldout_module==g)&(selected.task==t),'inner_selected_model'].iloc[0]
        assert chosen==expected
    if final:
        chosen=pd.read_csv(ROOT/'validation_selected_test_results.csv');assert len(chosen)==14
        assert len(res.loc[res.is_validation_selected])==14
        for _,r in chosen.iterrows():
            assert r.model==selected.loc[(selected.heldout_module==r.heldout_module)&(selected.task==r.task),'inner_selected_model'].iloc[0]
        combined=pd.concat([res,chosen.assign(model='Validation-selected')])
        macro=pd.read_csv(ROOT/'macro_domain_summary.csv');gap=pd.read_csv(ROOT/'generalization_gaps.csv')
        for _,r in macro.iterrows():
            z=combined.loc[(combined.task==r.task)&(combined.model==r.model)];assert len(z)==7
            for metric in ['AP','nAP','AUROC','Brier','Lift_at_10','Precision_at_10','Recall_at_10','F1']:
                for pref,val in [('Mean',z[metric].mean()),('SD',z[metric].std(ddof=0)),('Min',z[metric].min()),('Max',z[metric].max())]:close(r[f'{pref}_{metric}'],val)
        for _,r in gap.iterrows():
            ref=pd.read_csv(ROOT/f'E1_{r.task}_reference.csv').set_index('model').loc[r.E1_reference_model]
            z=combined.loc[(combined.task==r.task)&(combined.model==r.model)]
            close(r.Gap_AP,ref.AUPRC-z.AP.mean());close(r.Gap_nAP,(ref.AUPRC-ref.test_prevalence)/(1-ref.test_prevalence)-z.nAP.mean());close(r.Gap_AUROC,ref.AUROC-z.AUROC.mean())
    result=dict(status='passed',outer_task_folds=folds,inner_task_folds=inners,all_student_intersections_zero=True,frozen_files_verified=nfiles,feature_byte_identity=True,model_search_space_byte_identity=True,completed_trials_verified=trials,inner_domain_scores_verified=domain_scores,reloaded_models_verified=models)
    (ROOT/('final_independent_audit.json' if final else 'pretraining_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();audit(a.final)
