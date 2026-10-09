"""Frozen Day28 vs Day56, dynamic and common-record evaluation only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
import run_e4b as b
ROOT=b.ROOT;A=b.A
METRICS=['AP','nAP','AUROC','Brier']
def cached(y,p):
    p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=int);o=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[o])!=0),len(p)-1]
    return y,p,o,ends,(p-y)**2
def weighted(c,w):
    y,p,o,ends,sq=c;total=float(w.sum());positives=float(w@y)
    if positives==0 or positives==total:return np.full(4,np.nan)
    nn=np.cumsum(w[o])[ends];tp=np.cumsum((w*y)[o])[ends]
    precision=np.divide(tp,nn,out=np.zeros(len(tp)),where=nn>0);rec=tp/positives
    ap=float(np.sum(np.diff(np.r_[0.,rec])*precision));fpr=(nn-tp)/(total-positives)
    auc=float(np.sum((rec+np.r_[0.,rec[:-1]])*.5*np.diff(np.r_[0.,fpr])))
    pi=positives/total;return np.array([ap,(ap-pi)/(1-pi),auc,float(w@sq/total)])
def metrics(y,p):
    ap=average_precision_score(y,p);pi=np.mean(y)
    return dict(AP=ap,nAP=(ap-pi)/(1-pi),AUROC=roc_auc_score(y,p),Brier=brier_score_loss(y,p),N=len(y),Pos=int(np.sum(y)),Prevalence=pi)
def prediction(root,g,t,n):
    p=pd.read_csv(root/'folds'/g/t/n/'test_predictions.csv',float_precision='round_trip')
    return p[['enrolment_key','id_student','y','probability']]
def main():
    b.integrity();b.frozen_check();r28=pd.read_csv(A/'all_models_test_results.csv');r56=pd.read_csv(ROOT/'all_models_test_results.csv');s28=pd.read_csv(A/'inner_selected_models.csv');s56=pd.read_csv(ROOT/'inner_selected_models.csv')
    dyn=[];common=[];membership=[];bootstrapchecks=0
    for g in b.MOD:
        for task in b.TASKS:
            n28=s28.loc[(s28.heldout_module==g)&(s28.task==task),'inner_selected_model'].iloc[0];n56=s56.loc[(s56.heldout_module==g)&(s56.task==task),'inner_selected_model'].iloc[0]
            source=r28.loc[(r28.heldout_module==g)&(r28.task==task)&r28.model.isin(b.NAMES)]
            assert source.loc[source.inner_mean_AP.idxmax(),'model']==n28
            for label in b.NAMES+['Validation-selected']:
                a=n28 if label=='Validation-selected' else label;c=n56 if label=='Validation-selected' else label
                x=r28.loc[(r28.heldout_module==g)&(r28.task==task)&(r28.model==a)].iloc[0];z=r56.loc[(r56.heldout_module==g)&(r56.task==task)&(r56.model==c)].iloc[0]
                row=dict(heldout_module=g,task=task,comparison=label,Day28_model=a,Day56_model=c,Day28_N=x.test_N,Day56_N=z.test_N,Day28_Pos=x.test_Pos,Day56_Pos=z.test_Pos,Day28_Prevalence=x.test_Prevalence,Day56_Prevalence=z.test_Prevalence)
                for metric in METRICS:row.update({f'Day28_{metric}':x[metric],f'Day56_{metric}':z[metric],f'delta_{metric}':z[metric]-x[metric]})
                dyn.append(row)
                pa=prediction(A,g,task,a);pb=prediction(ROOT,g,task,c)
                joined=pa.merge(pb,on='enrolment_key',suffixes=('_28','_56'),validate='one_to_one').sort_values('enrolment_key').reset_index(drop=True)
                assert np.array_equal(joined.y_28,joined.y_56) and np.array_equal(joined.id_student_28,joined.id_student_56)
                y=joined.y_28.to_numpy();p=joined.probability_28.to_numpy();q=joined.probability_56.to_numpy();ids=joined.id_student_28.to_numpy()
                ma=metrics(y,p);mb=metrics(y,q);cr=dict(heldout_module=g,task=task,comparison=label,Day28_model=a,Day56_model=c,common_N=len(y),common_students=len(np.unique(ids)),common_Pos=int(y.sum()),common_Prevalence=y.mean())
                for metric in METRICS:cr.update({f'Day28_{metric}':ma[metric],f'Day56_{metric}':mb[metric],f'delta_{metric}':mb[metric]-ma[metric]})
                if label=='Validation-selected':
                    dest=ROOT/'common_cohort'/g/task;dest.mkdir(parents=True,exist_ok=True);joined.to_csv(dest/'paired_predictions.csv',index=False)
                    for key in joined.enrolment_key:membership.append(dict(heldout_module=g,task=task,enrolment_key=key))
                    ca=cached(y,p);cb=cached(y,q);_,inv=np.unique(ids,return_inverse=True);ns=inv.max()+1;rng=np.random.default_rng(b.e1.SEED);reps=[]
                    for i in range(1000):
                        weights=np.bincount(rng.integers(0,ns,ns),minlength=ns)[inv];v1=weighted(ca,weights);v2=weighted(cb,weights)
                        if i<5:
                            expanded=np.repeat(np.arange(len(y)),weights);checka=metrics(y[expanded],p[expanded]);checkb=metrics(y[expanded],q[expanded]);assert np.allclose(v1,[checka[k] for k in METRICS],rtol=1e-10,atol=1e-10);assert np.allclose(v2,[checkb[k] for k in METRICS],rtol=1e-10,atol=1e-10);bootstrapchecks+=1
                        reps.append([i,*(v2-v1)])
                    rep=pd.DataFrame(reps,columns=['replicate']+['delta_'+m for m in METRICS]);rep.to_csv(dest/'paired_bootstrap.csv',index=False)
                    cr['bootstrap_valid']=int(rep.delta_AP.notna().sum())
                    for metric in METRICS:
                        lo,hi=rep['delta_'+metric].dropna().quantile([.025,.975]);cr[f'delta_{metric}_CI_low']=lo;cr[f'delta_{metric}_CI_high']=hi
                common.append(cr)
    dynamic=pd.DataFrame(dyn);stable=pd.DataFrame(common)
    dynamic.to_csv(ROOT/'dynamic_horizon_comparison.csv',index=False);stable.to_csv(ROOT/'common_cohort_comparison.csv',index=False);pd.DataFrame(membership).to_csv(ROOT/'common_cohort_membership.csv',index=False)
    for task in b.TASKS:
        dynamic.loc[(dynamic.task==task)&(dynamic.comparison=='Validation-selected')].to_csv(ROOT/f'{task}_primary_horizon_gains.csv',index=False)
    rows=[]
    for cohort,data in [('dynamic',dynamic),('common',stable)]:
        for (task,label),z in data.groupby(['task','comparison']):
            row=dict(cohort=cohort,task=task,comparison=label,positive_nAP_modules=int(z.delta_nAP.gt(0).sum()))
            for metric in METRICS:
                row.update({f'Mean_Day28_{metric}':z[f'Day28_{metric}'].mean(),f'Mean_Day56_{metric}':z[f'Day56_{metric}'].mean(),f'Mean_delta_{metric}':z['delta_'+metric].mean(),f'SD_delta_{metric}':z['delta_'+metric].std(ddof=0),f'Min_delta_{metric}':z['delta_'+metric].min(),f'Max_delta_{metric}':z['delta_'+metric].max()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'horizon_gain_summary.csv',index=False)
    b.dump(ROOT/'comparison_checks.json',dict(status='passed',dynamic_comparisons=len(dynamic),common_comparisons=len(stable),selected_common_bootstrap_pairs=14,explicit_duplicate_replicates_verified=bootstrapchecks,Day28_selection_unchanged_by_three_family_restriction=True,no_refitting=True,frozen_files_unchanged=b.frozen_check()))
    print(dynamic.loc[dynamic.comparison=='Validation-selected',['heldout_module','task','Day28_model','Day56_model','Day28_nAP','Day56_nAP','delta_nAP']].to_string(index=False),flush=True)
if __name__=='__main__':main()
