"""Frozen source-only optimization engine for E4T and E5. No target scoring in workers."""
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
from pathlib import Path
import argparse,json,time,subprocess,concurrent.futures,gc
import numpy as np,pandas as pd,optuna,joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss,f1_score
import e1_reference as e1
ROOT=Path(__file__).resolve().parent
NAMES=['LR','XGBoost','CatBoost']
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,o):e1.dump(p,o)
def config():return load(ROOT/'config.json')
def data(c):return pd.read_csv(ROOT/c['data'],float_precision='round_trip').set_index('row_id',drop=False)
def members(exp,role):return pd.read_csv(ROOT/exp/(role+'.csv')).row_id.to_numpy()
def integrity():
    for p,h in load(ROOT/'run_manifest.json')['hashes'].items():assert e1.sha(ROOT/p)==h,p
def model(name,params,c):
    m=e1.model(name,params)
    cats=c.get('categorical',[])
    if not cats:return m
    num=[x for x in c['features'] if x not in cats]
    pre=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),('numeric','passthrough',num)],sparse_threshold=0)
    return make_pipeline(pre,m)
def threshold_curve(oof):
    cand=np.unique(np.r_[0.,oof.probability.to_numpy(),1.]);vals=[]
    for _,s in oof.groupby('validation_domain',sort=True):
        order=np.argsort(s.probability.to_numpy(),kind='stable');p=s.probability.to_numpy()[order];y=s.y.to_numpy()[order]
        pos=np.searchsorted(p,cand,side='left');cum=np.r_[0,np.cumsum(y)];tp=cum[-1]-cum[pos]
        vals.append(2*tp/np.maximum(y.sum()+len(y)-pos,1))
    mean=np.mean(vals,axis=0)
    return float(cand[np.argmax(mean)]),pd.DataFrame(dict(threshold=cand,mean_domain_F1=mean))
def metrics(y,p,keys,threshold=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pi=y.mean();ap=average_precision_score(y,p) if y.sum() else np.nan
    r=dict(N=len(y),Positive=int(y.sum()),Prevalence=pi,AP=ap,nAP=(ap-pi)/(1-pi) if 0<pi<1 else np.nan,AUROC=roc_auc_score(y,p) if 0<pi<1 else np.nan,Brier=brier_score_loss(y,p),LogLoss=log_loss(y,p,labels=[0,1]),F1=f1_score(y,p>=threshold,zero_division=0),threshold=threshold)
    order=np.lexsort((np.asarray(keys,dtype=str),-p))
    for pct in [5,10,20]:
        k=int(np.ceil(len(y)*pct/100));hit=int(y[order[:k]].sum());prec=hit/k
        r.update({f'Precision_at_{pct}':prec,f'Recall_at_{pct}':hit/y.sum() if y.sum() else np.nan,f'Lift_at_{pct}':prec/pi if pi else np.nan})
    return r
def preaudit():
    integrity();rows=[]
    for exp,c in config().items():
        d=data(c);tr=d.loc[members(exp,'train')];te=d.loc[members(exp,'test')]
        assert len(set(tr.id)&set(te.id))==0
        seen=[]
        for h in c['folds']:
            it=d.loc[members(exp,'inner_'+h+'_train')];iv=d.loc[members(exp,'inner_'+h+'_validation')]
            assert not set(it.id)&set(iv.id) and not set(it.id)&set(te.id) and not set(iv.id)&set(te.id)
            assert set(it.row_id)<=set(tr.row_id) and set(iv.row_id)<=set(d.loc[members(exp,'source')].row_id)
            assert it.y.nunique()==iv.y.nunique()==2
            if c['domain_disjoint']:assert not set(it.domain)&set(iv.domain)
            seen+=iv.row_id.tolist();rows.append(dict(experiment=exp,fold=h,train_N=len(it),validation_N=len(iv),train_Pos=int(it.y.sum()),validation_Pos=int(iv.y.sum()),student_overlap=0,domain_overlap=len(set(it.domain)&set(iv.domain))))
        assert len(seen)==len(set(seen))
        assert set(seen)==set(members(exp,c['validation_union']))
        assert set(c['features'])<=set(d) and d[c['features']].notna().all().all()
    pd.DataFrame(rows).to_csv(ROOT/'inner_split_audit.csv',index=False)
    dump(ROOT/'pretraining_audit.json',dict(status='passed',inner_folds=len(rows),experiments=len(config()),no_outer_or_inner_student_overlap=True))
def worker(exp,name):
    integrity();assert load(ROOT/'pretraining_audit.json')['status']=='passed'
    c=config()[exp];dest=ROOT/exp/name;dest.mkdir(exist_ok=True)
    if (dest/'selection.json').exists():return
    start=time.time();d=data(c);tr=d.loc[members(exp,'train')];parts=[]
    for h in c['folds']:parts.append((h,d.loc[members(exp,'inner_'+h+'_train')],d.loc[members(exp,'inner_'+h+'_validation')]))
    # Remove target rows from worker memory before optimization.
    del d
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=e1.SEED),storage='sqlite:///'+str((dest/'study.sqlite3').resolve()).replace('\\','/'),study_name=exp.replace('/','_')+'_'+name,load_if_exists=False)
    best=-np.inf
    def objective(t):
        nonlocal best
        params=e1.parameters(t,name);aps=[];domainrows=[];out=[]
        for h,it,iv in parts:
            m=model(name,params,c);m.fit(it[c['features']],it.y);p=m.predict_proba(iv[c['features']])[:,1].astype(float)
            ap=average_precision_score(iv.y,p);aps.append(ap)
            domainrows.append(dict(validation_domain=h,N=len(iv),AP=ap,nAP=(ap-iv.y.mean())/(1-iv.y.mean())))
            o=iv[['row_id','id','key','domain','module','y']].copy();o['validation_domain']=h;o['probability']=p;out.append(o)
        value=float(np.mean(aps));t.set_user_attr('domains',domainrows);t.set_user_attr('SD_AP',float(np.std(aps)));t.set_user_attr('Worst_AP',float(np.min(aps)))
        if value>best:
            best=value;pd.concat(out,ignore_index=True).to_csv(dest/'best_inner_predictions.csv.gz',index=False);dump(dest/'best_oof_trial.json',dict(trial=t.number))
        print(exp,name,'trial',t.number+1,'/',c['trials'],'AP',value,flush=True)
        return value
    study.optimize(objective,n_trials=c['trials']);study.trials_dataframe().to_csv(dest/'trials.csv',index=False)
    rows=[dict(trial=t.number,mean_AP=t.value,SD_AP=t.user_attrs['SD_AP'],Worst_AP=t.user_attrs['Worst_AP'],**s) for t in study.trials for s in t.user_attrs['domains']]
    pd.DataFrame(rows).to_csv(dest/'trial_inner_domains.csv',index=False);bt=study.best_trial
    assert load(dest/'best_oof_trial.json')['trial']==bt.number
    oof=pd.read_csv(dest/'best_inner_predictions.csv.gz',float_precision='round_trip');threshold,curve=threshold_curve(oof);curve.to_csv(dest/'threshold_curve.csv.gz',index=False)
    m=model(name,bt.params,c);m.fit(tr[c['features']],tr.y);joblib.dump(m,dest/'model.joblib',compress=3)
    dump(dest/'selection.json',dict(experiment=exp,model=name,params=bt.params,best_trial=bt.number,inner_mean_AP=bt.value,inner_SD_AP=bt.user_attrs['SD_AP'],inner_Worst_AP=bt.user_attrs['Worst_AP'],threshold=threshold,completed_trials=len(study.trials),runtime_seconds=time.time()-start,model_sha256=e1.sha(dest/'model.joblib')))
def train(workers):
    integrity();assert load(ROOT/'pretraining_audit.json')['status']=='passed'
    jobs=[(e,n) for e in config() for n in NAMES]
    def one(j):
        e,n=j
        if not (ROOT/e/n/'selection.json').exists():
            with (ROOT/e/(n+'_train.log')).open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-u',str(Path(__file__)),'worker','--experiment',e,'--model',n],stdout=f,stderr=subprocess.STDOUT,check=True)
        return j
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(one,j) for j in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):print('DONE',i,'/',len(jobs),f.result(),flush=True)
    choices=[];hashes={}
    for exp in config():
        ss=[load(ROOT/exp/n/'selection.json') for n in NAMES];s=max(ss,key=lambda z:z['inner_mean_AP']);choices.append(dict(experiment=exp,selected_model=s['model'],source_mean_AP=s['inner_mean_AP']))
        for n in NAMES:
            for fn in ['model.joblib','selection.json','best_inner_predictions.csv.gz']:hashes[exp+'/'+n+'/'+fn]=e1.sha(ROOT/exp/n/fn)
    pd.DataFrame(choices).to_csv(ROOT/'source_selected_models.csv',index=False);hashes['source_selected_models.csv']=e1.sha(ROOT/'source_selected_models.csv')
    dump(ROOT/'selection_lock.json',dict(locked_before_any_test_scoring=True,created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),hashes=hashes))
    dump(ROOT/'training_complete.json',dict(status='passed',studies=len(jobs),trials=sum(3*c['trials'] for c in config().values()),inner_fits=sum(3*c['trials']*len(c['folds']) for c in config().values()),final_fits=len(jobs)))
def score():
    integrity();lock=load(ROOT/'selection_lock.json')
    for p,h in lock['hashes'].items():assert e1.sha(ROOT/p)==h,p
    choices=pd.read_csv(ROOT/'source_selected_models.csv').set_index('experiment');rows=[];domains=[]
    for exp,c in config().items():
        d=data(c);te=d.loc[members(exp,'test')];tr=d.loc[members(exp,'train')]
        for n in NAMES:
            dest=ROOT/exp/n;s=load(dest/'selection.json');m=joblib.load(dest/'model.joblib');p=m.predict_proba(te[c['features']])[:,1].astype(float)
            pred=te[['row_id','id','key','domain','module','y']].copy();pred['probability']=p;pred.to_csv(dest/'test_predictions.csv',index=False)
            for scope,ix in [('all',np.ones(len(te),dtype=bool)),('matched_modules',te.module.isin(c.get('matched_modules',[])).to_numpy())]:
                if not ix.any():continue
                rows.append(dict(experiment=exp,scope=scope,model=n,selected=n==choices.loc[exp,'selected_model'],inner_mean_AP=s['inner_mean_AP'],source_prior=tr.y.mean(),source_prior_Brier=np.mean((te.y.to_numpy()[ix]-tr.y.mean())**2),**metrics(te.y.to_numpy()[ix],p[ix],te.key.to_numpy()[ix],s['threshold'])))
            for domain,z in pred.groupby('domain'):
                domains.append(dict(experiment=exp,model=n,selected=n==choices.loc[exp,'selected_model'],domain=domain,**metrics(z.y,z.probability,z.key,s['threshold'])))
            del m;gc.collect()
    r=pd.DataFrame(rows);r.to_csv(ROOT/'all_models_test_results.csv',index=False);r[r.selected].to_csv(ROOT/'validation_selected_test_results.csv',index=False);pd.DataFrame(domains).to_csv(ROOT/'test_domain_results.csv',index=False)
    dump(ROOT/'scoring_complete.json',dict(status='passed',test_rows=len(r),time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(r[r.selected].to_string(index=False),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['preaudit','train','worker','score']);a.add_argument('--workers',type=int,default=3);a.add_argument('--experiment');a.add_argument('--model');v=a.parse_args()
    if v.stage=='preaudit':preaudit()
    elif v.stage=='train':train(v.workers)
    elif v.stage=='worker':worker(v.experiment,v.model)
    else:score()
