"""Separate pretraining gate and final independent verification."""
import sys
sys.dont_write_bytecode=True
import argparse,json,zipfile,hashlib
from pathlib import Path
import pandas as pd,numpy as np,joblib
from scipy.stats import entropy
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve

KEY=['code_module','code_presentation','id_student'];TS=[14,28,56,84];TASKS=['withdrawal','failure'];NAMES=['LR','RF','XGBoost','LightGBM','CatBoost']
def key(d):return d.code_module.astype(str)+'|'+d.code_presentation.astype(str)+'|'+d.id_student.astype(str)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare_audit(oulad,base,out):
    manifest=load(out/'run_manifest.json');cols=manifest['features']
    master=pd.read_csv(out/'master_student_split.csv');mapping=master.set_index('id_student').split
    old=pd.read_csv(base/'oulad_day28_features.csv')
    assert np.array_equal(old.id_student.map(mapping),old.split)
    assert sha(out/'day28_features.csv')==sha(base/'oulad_day28_features.csv')
    assert sha(out/'e1_reference.py')==sha(base/'run_experiment.py')
    allgroups={s:set() for s in ['train','validation','test']};samples=[];counts=[]
    for t in TS:
        d=pd.read_csv(out/f'day{t}_features.csv');d['enrolment_key']=key(d)
        assert np.array_equal(d.id_student.map(mapping),d.split)
        assert not d.duplicated(KEY).any() and np.isfinite(d[cols]).all().all()
        audit=pd.read_csv(out/f'day{t}_cohort_audit.csv');inc=audit.loc[audit.cohort_status.eq('included')]
        assert inc.date_registration.notna().all() and inc.date_registration.le(t).all()
        assert not inc.date_unregistration.le(t).any()
        assert inc.loc[inc.final_result.eq('Withdrawn'),'date_unregistration'].gt(t).all()
        assert set(key(inc))==set(d.enrolment_key)
        for s in allgroups:allgroups[s]|=set(d.loc[d.split.eq(s),'id_student'])
        selected=pd.concat([d.sample(6,random_state=SEED+t),d.loc[d.total_clicks.eq(0)].head(1),d.loc[d.registration_lead_time.lt(0)].head(1)]).drop_duplicates('enrolment_key').assign(landmark=t)
        samples.append(selected)
        counts.append({'landmark':t,'records':len(d),'feature_values_finite':True,'active_cohort_valid':True})
    assert not (allgroups['train']&allgroups['test'] or allgroups['train']&allgroups['validation'] or allgroups['validation']&allgroups['test'])
    samples=pd.concat(samples,ignore_index=True)
    with zipfile.ZipFile(oulad) as z:
        sites=pd.read_csv(z.open('vle.csv'));chunks=[]
        for d in pd.read_csv(z.open('studentVle.csv'),chunksize=500000):chunks.append(d.loc[d.id_student.isin(samples.id_student)])
    logs=pd.concat(chunks,ignore_index=True)
    for _,r in samples.iterrows():
        t=int(r.landmark);start=max(0,int(-r.registration_lead_time))
        s=logs.loc[logs.id_student.eq(r.id_student)&logs.code_module.eq(r.code_module)&logs.code_presentation.eq(r.code_presentation)&logs.date.between(start,t)]
        daily=s.groupby('date').sum_click.sum().reindex(range(start,t+1),fill_value=0)
        active=int(daily.gt(0).sum());total=s.sum_click.sum()
        assert r.total_clicks==total and r.active_days==active
        assert np.isclose(r.active_ratio,active/len(daily))
        assert np.isclose(r.clicks_per_active_day,total/active if active else 0)
        assert np.isclose(r.daily_click_std,daily.std(ddof=0))
        assert r.unique_sites==s.id_site.nunique()
        recent=s.loc[s.date.between(t-6,t)].sum_click.sum();prev=s.loc[s.date.between(t-13,t-7)].sum_click.sum()
        assert r.recent7_clicks==recent and r.previous7_clicks==prev
        assert np.isclose(r.engagement_change,np.log1p(recent)-np.log1p(prev))
        assert r.inactivity_gap==(t-int(daily.loc[daily.gt(0)].index.max()) if active else len(daily))
        assert np.isclose(r.engagement_slope,np.polyfit(daily.index,daily.values,1)[0] if len(daily)>1 else 0,atol=1e-8)
        joined=s.merge(sites[['id_site','code_module','code_presentation','activity_type']],on=['id_site','code_module','code_presentation'],validate='many_to_one')
        assert r.unique_activity_types==joined.activity_type.nunique()
        en=entropy(joined.groupby('activity_type').sum_click.sum()) if len(joined) else 0
        assert np.isclose(r.activity_entropy,en)
    rows=load(out/'preparation_audit.json')['horizons']
    assert all(r['retained_log_max_day']<=r['landmark'] for r in rows)
    stable=pd.read_csv(out/'stable_cohort_membership.csv')
    for t in TS:
        d=pd.read_csv(out/f'day{t}_features.csv');assert set(stable.enrolment_key)<=set(key(d))
    result={'status':'passed','existing_student_split_unchanged':True,'cross_horizon_student_overlap':0,
        'max_log_day_within_cutoff':True,'exited_records_excluded':True,'E1_feature_code_byte_identical':True,'Day28_feature_file_byte_identical':True,
        'independent_raw_feature_checks':len(samples),'horizons':counts,'stable_membership_records':len(stable)}
    dump(out/'pretraining_independent_audit.json',result);print(json.dumps(result),flush=True)

def topk_independent(y,p,keys):
    d=pd.DataFrame({'y':y,'p':p,'key':keys}).sort_values(['p','key'],ascending=[False,True],kind='stable')
    ret={}
    for pct in [5,10,20]:
        n=int(np.ceil(len(d)*pct/100));hit=d.iloc[:n].y.sum()
        ret.update({f'Precision_at_{pct}':hit/n,f'Recall_at_{pct}':hit/d.y.sum(),f'Lift_at_{pct}':(hit/n)/d.y.mean()})
    return ret

def final_audit(base,out):
    manifest=load(out/'run_manifest.json');cols=manifest['features']
    allresults=pd.read_csv(out/'all_models_all_horizons.csv');assert len(allresults)==40 and allresults.completed_new_trials.sum()==900
    assert len(allresults.loc[allresults.landmark.ne(28)])==30
    assert allresults.loc[allresults.landmark.eq(28),'completed_new_trials'].eq(0).all()
    assert sha(out/'master_student_split.csv')==manifest['master_split_sha256']
    assert sha(out/'PROTOCOL.txt')==manifest['protocol_sha256']
    checked=0
    for t in TS:
        assert sha(out/f'day{t}_features.csv')==manifest['feature_hashes'][str(t)]
        d=pd.read_csv(out/f'day{t}_features.csv');d['enrolment_key']=key(d)
        for task in TASKS:
            rows=allresults.loc[allresults.landmark.eq(t)&allresults.task.eq(task)]
            chosen=load(out/f'day{t}'/f'{task}_selection.json')['model']
            assert chosen==rows.loc[rows.validation_AP.idxmax(),'model']
            test=d.loc[d.split.eq('test')&d['y_'+task].notna()]
            val=d.loc[d.split.eq('validation')&d['y_'+task].notna()]
            pred=pd.read_csv(out/f'day{t}'/f'{task}_test_predictions.csv')
            assert np.array_equal(pred.enrolment_key,test.enrolment_key)
            y=test['y_'+task].to_numpy(dtype=int)
            for _,r in rows.iterrows():
                modeldir=base/'models' if t==28 else out/f'day{t}'/'models'
                m=joblib.load(modeldir/f'{task}_{r.model}.joblib')
                p=m['model'].predict_proba(test[cols])[:,1]
                assert np.allclose(p,pred[r.model+'_probability'])
                assert np.isclose(average_precision_score(y,p),r.AP) and np.isclose(roc_auc_score(y,p),r.AUROC)
                assert np.isclose((r.AP-y.mean())/(1-y.mean()),r.nAP)
                vp=m['model'].predict_proba(val[cols])[:,1]
                assert np.isclose(average_precision_score(val['y_'+task],vp),r.validation_AP)
                prec,rec,thr=precision_recall_curve(val['y_'+task],vp);f=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-15)
                assert np.isclose(thr[np.argmax(f)],r.threshold)
                for name,value in topk_independent(y,p,test.enrolment_key.to_numpy()).items():assert np.isclose(r[name],value)
                if t!=28:
                    trials=pd.read_csv(out/f'day{t}'/'trials'/f'{task}_{r.model}.csv');assert (trials.state=='COMPLETE').sum()==30
                checked+=1
    stable=pd.read_csv(out/'stable_all_models_trajectory.csv')
    for task in TASKS:
        reference=None
        for t in TS:
            pred=pd.read_csv(out/f'day{t}'/f'{task}_stable_test_predictions.csv').sort_values('enrolment_key')
            if reference is None:reference=pred.enrolment_key.tolist()
            else:assert reference==pred.enrolment_key.tolist()
            for _,r in stable.loc[stable.landmark.eq(t)&stable.task.eq(task)].iterrows():
                y=pred['y_'+task].to_numpy();p=pred[r.model+'_probability'].to_numpy()
                assert np.isclose(average_precision_score(y,p),r.AP)
                assert np.isclose((r.AP-y.mean())/(1-y.mean()),r.nAP)
                for name,value in topk_independent(y,p,pred.enrolment_key.to_numpy()).items():assert np.isclose(r[name],value)
    student=pd.read_csv(out/'withdrawal_student_level_capacity.csv')
    for _,r in student.iterrows():
        pred=pd.read_csv(out/f'day{r.landmark}'/'withdrawal_test_predictions.csv')
        g=pred.groupby('id_student').agg(y=('y_withdrawal','max'),p=(r.model+'_probability','max')).reset_index()
        assert len(g)==r.N_students and g.y.sum()==r.Pos_students
        for name,value in topk_independent(g.y.to_numpy(),g.p.to_numpy(),g.id_student.map(lambda x:f'{x:012d}').to_numpy()).items():assert np.isclose(r[name],value)
    gains=pd.read_csv(out/'stable_paired_horizon_gains.csv')
    for _,r in gains.iterrows():
        b=pd.read_csv(out/f'stable_bootstrap_{r.task}_{r.from_day}_{r.to_day}.csv')
        for metric in ['AP','nAP']:
            lo,hi=np.quantile(b['delta_'+metric],[.025,.975]);assert np.isclose(lo,r[metric+'_CI_low']) and np.isclose(hi,r[metric+'_CI_high'])
        # Check first 10 paired bootstrap draws by physically duplicating enrolment rows.
        a=pd.read_csv(out/f'day{r.from_day}'/f'{r.task}_stable_test_predictions.csv').sort_values('enrolment_key')
        z=pd.read_csv(out/f'day{r.to_day}'/f'{r.task}_stable_test_predictions.csv').sort_values('enrolment_key')
        assert a.enrolment_key.tolist()==z.enrolment_key.tolist()
        y=a['y_'+r.task].to_numpy();pa=a[r.from_model+'_probability'].to_numpy();pz=z[r.to_model+'_probability'].to_numpy()
        _,inv=np.unique(a.id_student,return_inverse=True);n=inv.max()+1;rng=np.random.default_rng(SEED)
        for i in range(10):
            w=np.bincount(rng.integers(0,n,n),minlength=n)[inv];idx=np.repeat(np.arange(len(y)),w)
            delta=average_precision_score(y[idx],pz[idx])-average_precision_score(y[idx],pa[idx])
            assert np.isclose(delta,b.delta_AP.iloc[i]) and np.isclose(delta/(1-y[idx].mean()),b.delta_nAP.iloc[i])
    frozen=load(out/'frozen_sources.json');nfiles=0
    for item in frozen.values():
        for f,h in item['files'].items():assert sha(Path(item['directory'])/f)==h;nfiles+=1
    result={'status':'passed','models_checked':checked,'new_models':30,'new_trials':900,'Day28_not_retrained':True,
        'E1_E2_unchanged_files':nfiles,'AP_nAP_AUROC_thresholds_and_TopK_verified':True,'student_capacity_verified':True,'stable_test_rows_identical_across_horizons':True,'stable_paired_intervals_verified':True,'paired_replicates_checked_by_explicit_rows':60}
    dump(out/'final_independent_audit.json',result);print(json.dumps(result),flush=True)

SEED=20261007
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','final'],required=True);p.add_argument('--baseline',required=True);p.add_argument('--oulad');p.add_argument('--output',default=str(Path(__file__).parent));a=p.parse_args()
    if a.stage=='prepare':prepare_audit(a.oulad,Path(a.baseline),Path(a.output))
    else:final_audit(Path(a.baseline),Path(a.output))
