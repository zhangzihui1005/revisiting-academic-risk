"""Independent raw-record spot checks plus saved-model/result consistency checks."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve

def main():
    p=argparse.ArgumentParser();p.add_argument('--oulad',required=True);a=p.parse_args()
    out=Path(__file__).parent
    d=pd.read_csv(out/'oulad_day28_features.csv')
    manifest=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    cols=manifest['features']
    selected=pd.concat([d.sample(6,random_state=61),d.loc[d.total_clicks.eq(0)].head(1),d.loc[d.registration_lead_time.lt(0)].head(1)]).drop_duplicates('row_id')
    with zipfile.ZipFile(a.oulad) as z:
        v=pd.read_csv(z.open('vle.csv'))
        chunks=[]
        for c in pd.read_csv(z.open('studentVle.csv'),chunksize=500000):
            chunks.append(c.loc[c.id_student.isin(selected.id_student)])
    logs=pd.concat(chunks,ignore_index=True)
    for _,r in selected.iterrows():
        start=max(0,int(-r.registration_lead_time))
        s=logs.loc[logs.id_student.eq(r.id_student)&logs.code_module.eq(r.code_module)&logs.code_presentation.eq(r.code_presentation)&logs.date.between(start,28)]
        daily=s.groupby('date').sum_click.sum().reindex(range(start,29),fill_value=0)
        active=int(daily.gt(0).sum())
        assert r.total_clicks==s.sum_click.sum()
        assert r.active_days==active
        assert np.isclose(r.daily_click_std,daily.std(ddof=0))
        assert np.isclose(r.active_ratio,active/len(daily))
        assert r.unique_sites==s.id_site.nunique()
        assert r.recent7_clicks==s.loc[s.date.between(22,28)].sum_click.sum()
        assert r.previous7_clicks==s.loc[s.date.between(15,21)].sum_click.sum()
        gap=28-int(daily.loc[daily.gt(0)].index.max()) if active else len(daily)
        assert r.inactivity_gap==gap
        slope=np.polyfit(daily.index,daily.values,1)[0] if len(daily)>1 else 0
        assert np.isclose(r.engagement_slope,slope,atol=1e-9)
        joined=s.merge(v[['id_site','code_module','code_presentation','activity_type']],on=['id_site','code_module','code_presentation'],validate='many_to_one')
        en=entropy(joined.groupby('activity_type').sum_click.sum()) if len(joined) else 0
        assert np.isclose(r.activity_entropy,en)
    checked=[]
    for task in ['withdrawal','failure']:
        results=pd.read_csv(out/f'oulad_day28_{task}_results.csv')
        preds=pd.read_csv(out/f'{task}_test_predictions.csv')
        val=d.loc[d.split.eq('validation')&d['y_'+task].notna()]
        test=d.loc[d.split.eq('test')&d['y_'+task].notna()]
        assert np.array_equal(test.row_id.to_numpy(),preds.row_id.to_numpy())
        for _,r in results.iterrows():
            saved=joblib.load(out/'models'/f'{task}_{r.model}.joblib')
            vp=saved['model'].predict_proba(val[cols])[:,1]
            tp=saved['model'].predict_proba(test[cols])[:,1]
            assert np.allclose(tp,preds[r.model+'_probability'],rtol=1e-6,atol=1e-8)
            precision,recall,thresholds=precision_recall_curve(val['y_'+task],vp)
            f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
            threshold=thresholds[np.argmax(f)]
            assert np.isclose(threshold,r.threshold)
            assert np.isclose(average_precision_score(val['y_'+task],vp),r.validation_AUPRC)
            assert np.isclose(average_precision_score(test['y_'+task],tp),r.AUPRC)
            assert np.isclose(roc_auc_score(test['y_'+task],tp),r.AUROC)
            checked.append(task+'_'+r.model)
    checks={'status':'passed','raw_feature_records_checked':len(selected),'saved_models_checked':checked,
        'checks':['source log feature recomputation','model prediction matches saved probabilities','threshold chosen on validation','validation AP matches selected trial','test AP and AUROC match report']}
    (out/'independent_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks))

if __name__=='__main__':main()
