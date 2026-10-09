"""Locked-model comparisons; paired cluster bootstrap and publication figures."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as e
ROOT=Path(__file__).resolve().parent;METRICS=['AP','nAP','AUROC','Brier']
def cache(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);o=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[o])!=0),len(p)-1];return y,p,o,ends,(y-p)**2
def weighted(c,w):
    y,p,o,ends,sq=c;total=w.sum();pos=w@y
    if pos==0 or pos==total:return np.full(4,np.nan)
    nn=np.cumsum(w[o])[ends];tp=np.cumsum((w*y)[o])[ends];rec=tp/pos;prec=np.divide(tp,nn,out=np.zeros(len(tp)),where=nn>0);ap=np.diff(np.r_[0.,rec])@prec;pi=pos/total;fpr=(nn-tp)/(total-pos);auc=np.sum((rec+np.r_[0.,rec[:-1]])*.5*np.diff(np.r_[0.,fpr]));return np.array([ap,(ap-pi)/(1-pi),auc,w@sq/total])
def pair(pa,pb,dest):
    z=pa.merge(pb,on='key',suffixes=('_a','_b'),validate='one_to_one').sort_values('key').reset_index(drop=True);assert np.array_equal(z.y_a,z.y_b) and np.array_equal(z.id_a,z.id_b)
    y=z.y_a.to_numpy();p=z.probability_a.to_numpy();q=z.probability_b.to_numpy();a=cache(y,p);b=cache(y,q);_,inv=np.unique(z.id_a,return_inverse=True);n=inv.max()+1;rng=np.random.default_rng(e.e1.SEED);reps=[]
    pointa=weighted(a,np.ones(len(y)));pointb=weighted(b,np.ones(len(y)))
    for i in range(1000):
        w=np.bincount(rng.integers(0,n,n),minlength=n)[inv];va=weighted(a,w);vb=weighted(b,w);reps.append(vb-va)
        if i<3 and np.isfinite(va).all():
            idx=np.repeat(np.arange(len(y)),w)
            for pp,v in [(p,va),(q,vb)]:
                yy=y[idx];ap=average_precision_score(yy,pp[idx]);pi=yy.mean();assert np.allclose(v,[ap,(ap-pi)/(1-pi),roc_auc_score(yy,pp[idx]),brier_score_loss(yy,pp[idx])],atol=1e-10)
    dest.mkdir(parents=True,exist_ok=True);z.to_csv(dest/'paired_predictions.csv',index=False);rep=pd.DataFrame(reps,columns=['delta_'+m for m in METRICS]);rep.to_csv(dest/'paired_bootstrap.csv',index=False)
    r=dict(common_N=len(y),common_students=n,common_prevalence=y.mean(),bootstrap_valid=int(rep.delta_AP.notna().sum()))
    for k,m in enumerate(METRICS):
        lo,hi=rep['delta_'+m].dropna().quantile([.025,.975]);r.update({f'earlier_{m}':pointa[k],f'later_{m}':pointb[k],f'delta_{m}':pointb[k]-pointa[k],f'delta_{m}_CI_low':lo,f'delta_{m}_CI_high':hi})
    return r
def pred(exp,n):return pd.read_csv(ROOT/exp/n/'test_predictions.csv',float_precision='round_trip')
def savefig(fig,name):
    fig.savefig(ROOT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def main():
    e.integrity();assert e.load(ROOT/'final_audit.json')['status']=='passed';r=pd.read_csv(ROOT/'all_models_test_results.csv');choices=pd.read_csv(ROOT/'source_selected_models.csv').set_index('experiment');allrows=[]
    if ROOT.name.startswith('E4T'):
        a='Day28_failure';b='E4T-B_day56_failure_secondary';shared=e.config()[a]['matched_modules'];dyn=[]
        for scope in ['all','matched_modules']:
            for label in e.NAMES+['Validation-selected']:
                na=choices.loc[a,'selected_model'] if label=='Validation-selected' else label;nb=choices.loc[b,'selected_model'] if label=='Validation-selected' else label
                x=r[(r.experiment==a)&(r.scope==scope)&(r.model==na)].iloc[0];y=r[(r.experiment==b)&(r.scope==scope)&(r.model==nb)].iloc[0];row=dict(scope=scope,comparison=label,earlier_model=na,later_model=nb,earlier_N=x.N,later_N=y.N)
                for m in METRICS:row.update({f'earlier_{m}':x[m],f'later_{m}':y[m],f'delta_{m}':y[m]-x[m]})
                dyn.append(row);pa=pred(a,na);pb=pred(b,nb)
                if scope=='matched_modules':pa=pa[pa.module.isin(shared)];pb=pb[pb.module.isin(shared)]
                allrows.append(dict(scope=scope,comparison=label,earlier_model=na,later_model=nb,**pair(pa,pb,ROOT/'paired'/scope/label)))
        pd.DataFrame(dyn).to_csv(ROOT/'dynamic_horizon_comparison.csv',index=False)
        na=choices.loc[a,'selected_model'];nb=choices.loc[b,'selected_model'];pa=pred(a,na);pb=pred(b,nb);mods=[]
        for g in sorted(pa.module.unique()):mods.append(dict(module=g,**pair(pa[pa.module==g],pb[pb.module==g],ROOT/'paired_by_module'/g)))
        pd.DataFrame(mods).to_csv(ROOT/'selected_module_horizon_gains.csv',index=False)
        fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained');s=pd.DataFrame(mods);xx=np.arange(len(s));ax.errorbar(xx,s.delta_nAP,yerr=[s.delta_nAP-s.delta_nAP_CI_low,s.delta_nAP_CI_high-s.delta_nAP],fmt='o',capsize=4,color='#216a9c');ax.axhline(0,color='gray',ls='--');ax.set(xticks=xx,xticklabels=s.module,ylabel='Day56 − Day28 nAP',title='2014 Failure: common-cohort information gain\nSource-selected pipelines; paired 95% student-bootstrap intervals');ax.grid(axis='y',alpha=.2);savefig(fig,'future_year_failure_gain')
        dr=pd.read_csv(ROOT/'test_domain_results.csv');dr.groupby(['experiment','model'])[['AP','nAP','AUROC','Brier']].agg(['mean','std','min']).to_csv(ROOT/'equal_target_domain_summary.csv')
    elif ROOT.name.startswith('E5A'):
        for a,b in [('H0_strict','H1'),('H1','H2'),('H0_strict','H0_expanded')]:
            for label in e.NAMES+['Validation-selected']:
                na=choices.loc[a,'selected_model'] if label=='Validation-selected' else label;nb=choices.loc[b,'selected_model'] if label=='Validation-selected' else label
                allrows.append(dict(earlier=a,later=b,comparison=label,earlier_model=na,later_model=nb,**pair(pred(a,na),pred(b,nb),ROOT/'paired'/(a+'_to_'+b)/label)))
        fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained');hs=['H0_strict','H1','H2'];colors=['#22699a','#d48620','#4a926b']
        for n,col in zip(e.NAMES,colors):ax.plot(range(3),[r[(r.experiment==h)&(r.model==n)].nAP.iloc[0] for h in hs],marker='o',label=n,color=col)
        ax.plot(range(3),[r[(r.experiment==h)&r.selected].nAP.iloc[0] for h in hs],ls='--',marker='s',color='black',label='Validation-selected');ax.set(xticks=range(3),xticklabels=['H0 strict (21)','H1 (+6)','H2 (+6)'],ylabel='Test nAP',title='UCI: same 726 test students across information sets');ax.legend();ax.grid(alpha=.2);savefig(fig,'uci_information_horizons')
    else:
        sel=r[r.selected].assign(model='Validation-selected');allr=pd.concat([r,sel]);macro=[];rank=[]
        for scope in ['all_17','excluding_tiny_course_33']:
            z=allr if scope=='all_17' else allr[allr.experiment.ne('course_33')]
            for n,v in z.groupby('model'):
                row=dict(scope=scope,model=n,courses=len(v))
                for m in ['AP','nAP','AUROC','Brier','Lift_at_10']:row.update({f'Mean_{m}':v[m].mean(),f'SD_{m}':v[m].std(ddof=0),f'Min_{m}':v[m].min()})
                macro.append(row)
        pd.DataFrame(macro).to_csv(ROOT/'macro_course_summary.csv',index=False);pivot=r.pivot(index='experiment',columns='model',values='AP');ranks=pivot.rank(axis=1,ascending=False,method='average');ranks.to_csv(ROOT/'course_model_ranks.csv')
        for n in e.NAMES:rank.append(dict(model=n,mean_rank=ranks[n].mean(),first_place_count=int(ranks[n].eq(1).sum()),source_selected_count=int(choices.selected_model.eq(n).sum())))
        pd.DataFrame(rank).to_csv(ROOT/'rank_stability.csv',index=False)
        fig,ax=plt.subplots(figsize=(10,6),layout='constrained');pivot=r.pivot(index='experiment',columns='model',values='nAP').reindex(columns=e.NAMES);pivot=pivot.reindex(sorted(pivot.index,key=lambda s:int(s.split('_')[1])));im=ax.imshow(pivot.to_numpy(),aspect='auto',cmap='YlGnBu',vmin=min(0,pivot.min().min()),vmax=1)
        for i in range(len(pivot)):
            for j in range(3):ax.text(j,i,f'{pivot.iloc[i,j]:.3f}',ha='center',va='center',color='white' if pivot.iloc[i,j]>.6 else 'black',fontsize=9)
        ax.set(xticks=range(3),xticklabels=e.NAMES,yticks=range(len(pivot)),yticklabels=[s.replace('course_','')+(' * N=9' if s=='course_33' else '') for s in pivot.index],ylabel='Held-out UCI course',title='UCI H1: unseen-course nAP (17 outer folds)');fig.colorbar(im,ax=ax,label='nAP');savefig(fig,'uci_loco_nap')
        pooled=[]
        for n in e.NAMES+['Validation-selected']:
            p=pd.concat([pred(ex,choices.loc[ex,'selected_model'] if n=='Validation-selected' else n) for ex in e.config()]);pooled.append(dict(model=n,**e.metrics(p.y,p.probability,p.key)))
        pd.DataFrame(pooled).to_csv(ROOT/'pooled_cross_course_predictions_summary.csv',index=False)
    if allrows:pd.DataFrame(allrows).to_csv(ROOT/'paired_horizon_comparison.csv',index=False)
    module_pairs=7 if ROOT.name.startswith('E4T') else 0
    e.dump(ROOT/'analysis_checks.json',dict(status='passed',paired_comparisons=len(allrows)+module_pairs,main_paired_comparisons=len(allrows),module_paired_comparisons=module_pairs,bootstrap_repeats=1000 if allrows else 0,first_three_replicates_verified_with_explicit_duplication=bool(allrows)))
    print('ANALYSIS COMPLETE',ROOT.name,flush=True)
if __name__=='__main__':main()
