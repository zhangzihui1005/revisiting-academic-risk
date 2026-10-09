"""Independent artifact, selection, membership, prediction and source-refit audit."""
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
from pathlib import Path
import argparse,json,hashlib,time,zipfile
import numpy as np,pandas as pd,joblib
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,f1_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));import run as e
def close(a,b):assert np.allclose(a,b,atol=1e-10,rtol=1e-10,equal_nan=True),(a,b)
def main(refit=True):
    e.integrity();e.preaudit();cfg=e.config();choices=pd.read_csv(ROOT/'source_selected_models.csv').set_index('experiment');results=pd.read_csv(ROOT/'all_models_test_results.csv');checks=[];refits=0
    lock=e.load(ROOT/'selection_lock.json');assert lock['locked_before_any_test_scoring']
    for p,h in lock['hashes'].items():assert e.e1.sha(ROOT/p)==h,p
    for exp,c in cfg.items():
        d=e.data(c);tr=d.loc[e.members(exp,'train')];te=d.loc[e.members(exp,'test')];expected=[]
        for n in e.NAMES:
            dest=ROOT/exp/n;s=e.load(dest/'selection.json');expected.append(s);t=pd.read_csv(dest/'trials.csv');domains=pd.read_csv(dest/'trial_inner_domains.csv');assert len(t)==c['trials'] and (t.state=='COMPLETE').all()
            group=domains.groupby('trial').AP;close(group.mean(),t.sort_values('number').value);close(group.std(ddof=0),domains.groupby('trial').SD_AP.first());close(group.min(),domains.groupby('trial').Worst_AP.first());assert domains.groupby('trial').size().eq(len(c['folds'])).all()
            assert int(t.loc[t.value.idxmax(),'number'])==s['best_trial'];close(t.value.max(),s['inner_mean_AP']);assert e.e1.sha(dest/'model.joblib')==s['model_sha256']
            oof=pd.read_csv(dest/'best_inner_predictions.csv.gz',float_precision='round_trip').set_index('row_id',drop=False);aps=[]
            for g in c['folds']:
                it=d.loc[e.members(exp,'inner_'+g+'_train')];iv=d.loc[e.members(exp,'inner_'+g+'_validation')];oo=oof.loc[iv.row_id]
                assert oo.validation_domain.astype(str).eq(g).all();assert np.array_equal(oo.y,iv.y);aps.append(average_precision_score(oo.y,oo.probability))
                if refit:
                    m=e.model(n,s['params'],c);m.fit(it[c['features']],it.y);p=m.predict_proba(iv[c['features']])[:,1].astype(float);close(p,oo.probability);refits+=1
            close(np.mean(aps),s['inner_mean_AP']);curve=pd.read_csv(dest/'threshold_curve.csv.gz',float_precision='round_trip');threshold=float(curve.loc[curve.mean_domain_F1.idxmax(),'threshold']);close(threshold,s['threshold'])
            for j in np.unique(np.r_[np.linspace(0,len(curve)-1,9,dtype=int),curve.mean_domain_F1.idxmax()]):
                z=curve.iloc[j];f=np.mean([f1_score(o.y,o.probability>=z.threshold,zero_division=0) for _,o in oof.groupby('validation_domain')]);close(f,z.mean_domain_F1)
            m=joblib.load(dest/'model.joblib');p=m.predict_proba(te[c['features']])[:,1].astype(float);saved=pd.read_csv(dest/'test_predictions.csv',float_precision='round_trip');assert np.array_equal(saved.row_id,te.row_id);assert np.array_equal(saved.y,te.y);close(p,saved.probability)
            if c.get('categorical'):
                pre=m.steps[0][1];enc=pre.named_transformers_['categorical']
                for f,levels in zip(c['categorical'],enc.categories_):assert np.array_equal(np.sort(tr[f].unique()),levels),f
                if n=='LR':close(m.steps[-1][1].steps[0][1].mean_,pre.transform(tr[c['features']]).mean(axis=0))
            for _,row in results[(results.experiment==exp)&(results.model==n)].iterrows():
                mask=np.ones(len(te),dtype=bool) if row.scope=='all' else te.module.isin(c['matched_modules']).to_numpy();y=te.y.to_numpy()[mask];pp=p[mask];pi=y.mean();ap=average_precision_score(y,pp)
                close([row.AP,row.nAP,row.AUROC,row.Brier],[ap,(ap-pi)/(1-pi),roc_auc_score(y,pp),brier_score_loss(y,pp)])
                order=np.lexsort((te.key.to_numpy()[mask],-pp))
                for pct in [5,10,20]:close(row[f'Lift_at_{pct}'],y[order[:int(np.ceil(len(y)*pct/100))]].mean()/pi)
            checks.append(dict(experiment=exp,model=n,trials=len(t),best_trial=s['best_trial'],source_mean_AP=s['inner_mean_AP'],OOF_refits=len(c['folds']) if refit else 0,model_predictions_verified=True))
            print('AUDITED',exp,n,flush=True)
        assert max(expected,key=lambda s:s['inner_mean_AP'])['model']==choices.loc[exp,'selected_model']
    oldcount=0
    if (ROOT/'historical_source_hashes.json').exists():
        for dn,files in e.load(ROOT/'historical_source_hashes.json').items():
            for rel,h in files.items():assert e.e1.sha(ROOT.parent/dn/rel)==h,(dn,rel);oldcount+=1
    pd.DataFrame(checks).to_csv(ROOT/'model_audit.csv',index=False);e.dump(ROOT/'final_audit.json',dict(status='passed',model_count=len(checks),trials=sum(z['trials'] for z in checks),source_best_trial_models_independently_refitted=refits,all_saved_predictions_verified=True,all_family_choices_source_only=True,historical_files_unchanged=oldcount,finished=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
if __name__=='__main__':main()
