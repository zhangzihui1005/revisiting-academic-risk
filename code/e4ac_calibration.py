import sys,os
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6')
from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd
from scipy.optimize import minimize,brentq
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'E4A_day28_student_disjoint_LOMO_v1.0'
SEED=20261007;EPS=1e-12
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def logits(p):return logit(np.clip(np.asarray(p,dtype=float),EPS,1-EPS))
def fit(y,p,increasing=True):
    y=np.asarray(y,dtype=float);z=logits(p)
    def fun(c):
        eta=c[0]+c[1]*z;q=expit(eta);r=q-y
        return np.mean(np.logaddexp(0,eta)-y*eta),np.array([r.mean(),np.mean(r*z)])
    initial=np.array([logit(y.mean())-z.mean(),1.])
    opt=minimize(fun,initial,jac=True,method='L-BFGS-B',bounds=[(None,None),(1e-8,None)] if increasing else None,options={'ftol':1e-14,'gtol':1e-9,'maxiter':10000})
    assert opt.success,(opt.message,opt.x)
    grad=fun(opt.x)[1];assert np.max(abs(grad))<1e-6 or (increasing and opt.x[1]<=1.01e-8)
    return dict(intercept=float(opt.x[0]),slope=float(opt.x[1]),converged=bool(opt.success),gradient_max=float(np.max(abs(grad))),slope_on_boundary=bool(increasing and opt.x[1]<=1.01e-8),objective=float(opt.fun))
def transform(p,c):return expit(c['intercept']+c['slope']*logits(p))
def evaluate(y,p,constant=False):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pi=y.mean();ap=average_precision_score(y,p)
    diag=fit(y,p,False) if not constant else {'intercept':None,'slope':None}
    citl=brentq(lambda a:np.mean(expit(a+logits(p)))-pi,-100,100)
    return dict(N=len(y),Pos=int(y.sum()),Prevalence=pi,AP=ap,nAP=(ap-pi)/(1-pi),AUROC=roc_auc_score(y,p),Brier=brier_score_loss(y,p),LogLoss=log_loss(y,np.clip(p,EPS,1-EPS),labels=[0,1]),Calibration_intercept=diag['intercept'],Calibration_slope=diag['slope'],CITL_intercept=citl)
def paired(y,p,q,ids):
    y=np.asarray(y,dtype=float);p=np.clip(p,EPS,1-EPS);q=np.clip(q,EPS,1-EPS)
    _,inv=np.unique(ids,return_inverse=True);n=inv.max()+1;counts=np.bincount(inv,minlength=n)
    db=(q-y)**2-(p-y)**2;dl=-(y*np.log(q)+(1-y)*np.log1p(-q))+(y*np.log(p)+(1-y)*np.log1p(-p))
    sums=np.vstack([np.bincount(inv,weights=v,minlength=n) for v in [db,dl]])
    rng=np.random.default_rng(SEED);vals=[]
    for i in range(1000):
        w=np.bincount(rng.integers(0,n,n),minlength=n);v=sums@w/(counts@w);vals.append((i,*v))
    return pd.DataFrame(vals,columns=['replicate','delta_Brier','delta_LogLoss'])
def bins(y,p,w=None):
    y=np.asarray(y);p=np.asarray(p);w=np.ones(len(y)) if w is None else np.asarray(w)
    codes=np.minimum((p*10).astype(int),9);rows=[]
    for b in range(10):
        keep=codes==b
        if not keep.any():continue
        rows.append(dict(bin=b,lower=b/10,upper=(b+1)/10,N=int(keep.sum()),weight_sum=float(w[keep].sum()),mean_probability=float(np.average(p[keep],weights=w[keep])),observed_rate=float(np.average(y[keep],weights=w[keep]))))
    return rows
def main():
    assert not (ROOT/'calibrators_locked.json').exists(),'Use a new output directory to rerun.'
    history={str(p.relative_to(SOURCE)).replace('\\','/'):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    dump(ROOT/'frozen_source_manifest.json',dict(directory=str(SOURCE),files=history))
    dump(ROOT/'run_manifest.json',dict(seed=SEED,protocol_sha256=sha(ROOT/'PROTOCOL.txt'),code_sha256=sha(Path(__file__)),source=SOURCE.name))
    shutil.copy2(SOURCE/'requirements-lock.txt',ROOT/'requirements-lock.txt')
    selected=pd.read_csv(SOURCE/'inner_selected_models.csv');assert len(selected)==14 and selected.inner_selected_model.eq('LR').all()
    audits=[];calibrators=[]
    for row in selected.itertuples():
        g=row.heldout_module;t=row.task;base=SOURCE/'folds'/g/t;dest=ROOT/'folds'/g/t;dest.mkdir(parents=True)
        o=pd.read_csv(base/'LR/best_inner_predictions.csv.gz',float_precision='round_trip');members=pd.read_csv(base/'train_membership.csv');held=set(pd.read_csv(base.parent/'heldout_students.csv').id_student)
        assert set(o.row_id)==set(members.row_id) and len(o)==len(members) and not o.row_id.duplicated().any()
        joined=o.merge(members[['row_id','y_'+t]],on='row_id',validate='one_to_one');assert np.array_equal(joined.y,joined['y_'+t])
        assert g not in set(o.code_module) and o.code_module.nunique()==6 and set(o.id_student).isdisjoint(held)
        for h in sorted(o.code_module.unique()):
            vi=pd.read_csv(base/f'inner_{h}_validation_membership.csv');tr=pd.read_csv(base/f'inner_{h}_train_membership.csv')
            assert set(o.loc[o.code_module==h,'row_id'])==set(vi.row_id)
            assert set(tr.id_student).isdisjoint(set(vi.id_student)) and set(tr.id_student).isdisjoint(held)
        c=fit(o.y,o.probability);dump(dest/'calibrator.json',c)
        o.to_csv(dest/'source_oof_predictions.csv.gz',index=False)
        audits.append(dict(heldout_module=g,task=t,source_N=len(o),source_students=o.id_student.nunique(),source_domains=6,source_heldout_student_overlap=0,source_prevalence=o.y.mean(),**c))
        calibrators.append(dict(heldout_module=g,task=t,path=str((dest/'calibrator.json').relative_to(ROOT)),sha256=sha(dest/'calibrator.json')))
    pd.DataFrame(audits).to_csv(ROOT/'source_calibration_audit.csv',index=False)
    dump(ROOT/'calibrators_locked.json',dict(calibrators=calibrators,all_fits_completed_before_target_evaluation=True))
    print('SOURCE_CALIBRATORS_LOCKED 14',flush=True)
    allresults=[];changes=[];binrows=[];preds=[]
    for item in calibrators:
        g=item['heldout_module'];t=item['task'];dest=ROOT/'folds'/g/t
        assert sha(ROOT/item['path'])==item['sha256'];c=load(ROOT/item['path'])
        p=pd.read_csv(SOURCE/'folds'/g/t/'LR/test_predictions.csv',float_precision='round_trip');prior=pd.read_csv(SOURCE/'folds'/g/t/'train_membership.csv')['y_'+t].mean()
        p=p.rename(columns={'probability':'raw'});p['calibrated']=transform(p.raw,c);p['source_prior']=prior;p['task']=t
        p.to_csv(dest/'test_predictions.csv',index=False);preds.append(p)
        rr={}
        for method in ['raw','calibrated','source_prior']:
            rr[method]=evaluate(p.y,p[method],method=='source_prior');allresults.append(dict(heldout_module=g,task=t,method=method,**rr[method]))
            for b in bins(p.y,p[method]):binrows.append(dict(task=t,module=g,method=method,**b))
        assert np.isclose(rr['raw']['AP'],rr['calibrated']['AP'],atol=1e-12) and np.isclose(rr['raw']['AUROC'],rr['calibrated']['AUROC'],atol=1e-12)
        reps=paired(p.y,p.raw.to_numpy(),p.calibrated.to_numpy(),p.id_student);reps.to_csv(dest/'paired_bootstrap.csv',index=False)
        change=dict(heldout_module=g,task=t,delta_AP=rr['calibrated']['AP']-rr['raw']['AP'],delta_AUROC=rr['calibrated']['AUROC']-rr['raw']['AUROC'])
        for metric in ['Brier','LogLoss']:
            ci=np.quantile(reps['delta_'+metric],[.025,.975]);change.update({f'delta_{metric}':rr['calibrated'][metric]-rr['raw'][metric],f'delta_{metric}_CI_low':ci[0],f'delta_{metric}_CI_high':ci[1]})
        changes.append(change)
    allr=pd.DataFrame(allresults);allr.to_csv(ROOT/'calibration_results.csv',index=False);pd.DataFrame(changes).to_csv(ROOT/'paired_calibration_changes.csv',index=False)
    macro=[]
    for (t,method),z in allr.groupby(['task','method']):
        row=dict(task=t,method=method)
        for metric in ['AP','nAP','AUROC','Brier','LogLoss','Calibration_intercept','Calibration_slope','CITL_intercept']:
            row.update({f'Mean_{metric}':z[metric].mean(),f'SD_{metric}':z[metric].std(ddof=0),f'Min_{metric}':z[metric].min(),f'Max_{metric}':z[metric].max()})
        macro.append(row)
    pd.DataFrame(macro).to_csv(ROOT/'macro_calibration_results.csv',index=False)
    pred=pd.concat(preds,ignore_index=True)
    for t,z in pred.groupby('task'):
        w=1/z.groupby('code_module').code_module.transform('size')
        for method in ['raw','calibrated','source_prior']:
            for b in bins(z.y,z[method],w):binrows.append(dict(task=t,module='Equal-domain pooled',method=method,**b))
    pd.DataFrame(binrows).to_csv(ROOT/'reliability_bins.csv',index=False)
    for rel,h in history.items():assert sha(SOURCE/rel)==h,rel
    dump(ROOT/'execution_checks.json',dict(status='passed',calibrators=14,base_models_retrained=0,source_test_overlap=0,AP_AUROC_preserved=True,historical_files_unchanged=len(history)))
    print(pd.DataFrame(macro)[['task','method','Mean_AP','Mean_AUROC','Mean_Brier','Mean_LogLoss']].to_string(index=False),flush=True)
if __name__=='__main__':main()
