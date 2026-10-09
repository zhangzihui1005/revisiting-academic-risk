"""Source-only increasing Platt calibration, selected frozen base pipelines."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import time,shutil
import numpy as np,pandas as pd
from scipy.optimize import minimize,brentq
from scipy.special import expit,logit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import e6_common as c
ROOT=Path(__file__).resolve().parent;EPS=1e-12
def logits(p):return logit(np.clip(np.asarray(p,dtype=float),EPS,1-EPS))
def fit(y,p):
 y=np.asarray(y,dtype=float);z=logits(p)
 def fun(v):
  eta=v[0]+v[1]*z;q=expit(eta);return np.mean(np.logaddexp(0,eta)-y*eta),np.array([(q-y).mean(),np.mean((q-y)*z)])
 op=minimize(fun,[logit(y.mean())-z.mean(),1.],jac=True,method='L-BFGS-B',bounds=[(None,None),(1e-8,None)],options=dict(ftol=1e-14,gtol=1e-9,maxiter=10000));assert op.success;g=fun(op.x)[1];assert np.max(abs(g))<1e-6 or op.x[1]<1.01e-8
 return dict(intercept=float(op.x[0]),slope=float(op.x[1]),gradient_max=float(np.max(abs(g))),slope_on_boundary=bool(op.x[1]<1.01e-8))
def trans(p,v):
 # Restore strictly increasing order if floating-point logit/expit collapses
 # distinct input probabilities. Identical raw probabilities remain identical.
 raw=np.asarray(p,dtype=float);u,inv=np.unique(raw,return_inverse=True);q=expit(v['intercept']+v['slope']*logits(u))
 for i in range(1,len(q)):
  if q[i]<=q[i-1]:q[i]=np.nextafter(q[i-1],1.)
 for i in range(len(q)-2,-1,-1):
  if q[i]>=q[i+1]:q[i]=np.nextafter(q[i+1],0.)
 assert np.all(np.diff(q)>0);return q[inv]
def diagnostic(y,p,constant=False):
 y=np.asarray(y);z=logits(p);citl=brentq(lambda a:expit(a+z).mean()-y.mean(),-100,100)
 if constant:return dict(calibration_intercept=np.nan,calibration_slope=np.nan,CITL=citl,diagnostic_status='constant: joint slope unidentified')
 if np.std(z)<1e-8:return dict(calibration_intercept=np.nan,calibration_slope=np.nan,CITL=citl,diagnostic_status='numerically near-constant: joint slope unidentified')
 if max(z[y==0])<min(z[y==1]) or max(z[y==1])<min(z[y==0]):return dict(calibration_intercept=np.nan,calibration_slope=np.nan,CITL=citl,diagnostic_status='complete separation: no finite joint MLE')
 def fun(v):
  eta=v[0]+v[1]*z;q=expit(eta);return np.mean(np.logaddexp(0,eta)-y*eta),np.array([(q-y).mean(),np.mean((q-y)*z)])
 op=minimize(fun,[0.,1.],jac=True,method='BFGS',options=dict(gtol=1e-8,maxiter=10000));grad=np.max(abs(fun(op.x)[1]));valid=grad<1e-6
 return dict(calibration_intercept=float(op.x[0]) if valid else np.nan,calibration_slope=float(op.x[1]) if valid else np.nan,CITL=citl,diagnostic_status='finite joint MLE' if valid else 'optimizer/identifiability failure')
def bins(y,p):
 y=np.asarray(y);p=np.asarray(p);codes=np.minimum((p*10).astype(int),9);out=[]
 for b in range(10):
  keep=codes==b
  if keep.any():out.append(dict(bin=b,N=int(keep.sum()),mean_probability=p[keep].mean(),observed_rate=y[keep].mean()))
 return out
def metrics(y,p,constant=False):
 r=c.metrics(y,p);bb=bins(y,p);r['ECE']=sum(z['N']*abs(z['observed_rate']-z['mean_probability']) for z in bb)/len(y);r.update(diagnostic(y,p,constant));return r
def bootstrap(z):
 y=z.y.to_numpy();p=np.clip(z.raw.to_numpy(),EPS,1-EPS);q=np.clip(z.calibrated.to_numpy(),EPS,1-EPS);_,inv=np.unique(z.id,return_inverse=True);n=inv.max()+1;count=np.bincount(inv);db=(q-y)**2-(p-y)**2;dl=-(y*np.log(q)+(1-y)*np.log1p(-q))+(y*np.log(p)+(1-y)*np.log1p(-p));sums=np.vstack([np.bincount(inv,weights=v,minlength=n) for v in [db,dl]]);rng=np.random.default_rng(c.SEED);out=[]
 for i in range(1000):
  w=np.bincount(rng.integers(0,n,n),minlength=n);out.append(sums@w/(count@w))
 return pd.DataFrame(out,columns=['delta_Brier','delta_LogLoss'])
def main():
 specs=c.specs(ROOT.parent,True);locked=[];audit=[]
 reuse=(ROOT/'calibrators_locked.json').exists()
 if reuse:locked=c.load(ROOT/'calibrators_locked.json')['specs']
 for s in ([] if reuse else specs):
  dest=ROOT/s['key'];dest.mkdir();o=c.aligned(s,'source');p=o[s['selected']].to_numpy()
  if s['environment']=='A':
   prev=ROOT.parent/'E4A_C_source_only_calibration_v1.0/folds'/s['outer']/s['task']/'calibrator.json';v=c.load(prev);v['reused_from']=str(prev.relative_to(ROOT.parent));fresh=fit(o.y,p);assert abs(fresh['intercept']-v['intercept'])<1e-5 and abs(fresh['slope']-v['slope'])<1e-5
  else:v=fit(o.y,p)
  c.dump(dest/'calibrator.json',v);o['raw']=p;o['calibrated']=trans(p,v);o.to_csv(dest/'source_predictions.csv.gz',index=False);locked.append(dict(**s,calibrator_sha256=c.sha(dest/'calibrator.json'),source_sha256=c.sha(dest/'source_predictions.csv.gz')));audit.append(dict(experiment=s['key'],setting=s['setting'],N=len(o),model=s['selected'],reused=s['environment']=='A',**v))
 if not reuse:
  pd.DataFrame(audit).to_csv(ROOT/'source_calibrator_audit.csv',index=False);c.dump(ROOT/'calibrators_locked.json',dict(specs=locked,all_fits_precede_target_scoring=True,code_sha256=c.sha(Path(__file__)),protocol_sha256=c.sha(ROOT/'PROTOCOL.txt')))
 results=[];changes=[];binrows=[];source_rows=[];rankchecks=[]
 for s in locked:
  dest=ROOT/s['key'];assert c.sha(dest/'calibrator.json')==s['calibrator_sha256'];v=c.load(dest/'calibrator.json');o=pd.read_csv(dest/'source_predictions.csv.gz',float_precision='round_trip');assert c.sha(dest/'source_predictions.csv.gz')==s['source_sha256'];o['calibrated']=trans(o.raw,v);z=c.aligned(s,'target');assert not set(z.id)&set(o.id);z['raw']=z[s['selected']];z['calibrated']=trans(z.raw,v);z['source_prior']=o.y.mean();z.to_csv(dest/'target_predictions.csv',index=False)
  for name in ['raw','calibrated']:
   source_rows.append(dict(experiment=s['key'],setting=s['setting'],method=name,**metrics(o.y,o[name])))
  rr={}
  for name in ['raw','calibrated','source_prior']:
   rr[name]=metrics(z.y,z[name],name=='source_prior');results.append(dict(experiment=s['key'],setting=s['setting'],outer=s['outer'],base_model=s['selected'],method=name,**rr[name]))
   binrows += [dict(experiment=s['key'],setting=s['setting'],method=name,**b) for b in bins(z.y,z[name])]
  da=rr['calibrated']['AP']-rr['raw']['AP'];du=rr['calibrated']['AUROC']-rr['raw']['AUROC'];assert abs(da)<1e-12 and abs(du)<1e-12,(s['key'],da,du);plain=expit(v['intercept']+v['slope']*logits(z.raw));adjustment=np.abs(z.calibrated.to_numpy()-plain);assert adjustment.max()<1e-12;rankchecks.append(dict(experiment=s['key'],delta_AP=da,delta_AUROC=du,numeric_tie_adjusted_rows=int((adjustment>0).sum()),max_numeric_adjustment=adjustment.max()))
  reps=bootstrap(z);reps.to_csv(dest/'paired_bootstrap.csv',index=False);cr=dict(experiment=s['key'],setting=s['setting'],outer=s['outer'])
  for m in ['Brier','LogLoss']:
   lo,hi=reps['delta_'+m].quantile([.025,.975]);cr.update({f'delta_{m}':rr['calibrated'][m]-rr['raw'][m],f'delta_{m}_CI_low':lo,f'delta_{m}_CI_high':hi})
  changes.append(cr);print('CALIBRATED',s['key'],flush=True)
 pd.DataFrame(results).to_csv(ROOT/'target_calibration_metrics.csv',index=False);pd.DataFrame(source_rows).to_csv(ROOT/'source_apparent_calibration_metrics.csv',index=False);pd.DataFrame(changes).to_csv(ROOT/'paired_calibration_changes.csv',index=False);pd.DataFrame(binrows).to_csv(ROOT/'reliability_bins.csv',index=False);pd.DataFrame(rankchecks).to_csv(ROOT/'ranking_sanity_checks.csv',index=False)
 r=pd.DataFrame(results);summary=[]
 for (setting,method),z in r.groupby(['setting','method']):
  row=dict(setting=setting,method=method,outer_evaluations=len(z))
  for m in ['AP','nAP','AUROC','Brier','LogLoss','ECE']:row.update({f'Mean_{m}':z[m].mean(),f'SD_{m}':z[m].std(ddof=0),f'Max_{m}':z[m].max(),f'Min_{m}':z[m].min()})
  summary.append(row)
 pd.DataFrame(summary).to_csv(ROOT/'calibration_summary.csv',index=False)
 bb=pd.DataFrame(binrows)
 for setting,z in bb.groupby('setting'):
  keys=list(z.experiment.unique());cols=min(3,len(keys));nr=int(np.ceil(len(keys)/cols));fig,axes=plt.subplots(nr,cols,figsize=(4.1*cols,3.2*nr),squeeze=False,layout='constrained')
  for ax,key in zip(axes.flat,keys):
   ax.plot([0,1],[0,1],ls='--',color='gray',lw=.8)
   for method,col in [('raw','#a15b32'),('calibrated','#24759c')]:
    q=z[(z.experiment==key)&z.method.eq(method)];ax.plot(q.mean_probability,q.observed_rate,marker='o',ms=3,label=method,color=col);sparse=q[q.N<20];ax.scatter(sparse.mean_probability,sparse.observed_rate,facecolor='white',edgecolor=col,zorder=4)
   ax.set(xlim=(0,1),ylim=(0,1),title=key,xlabel='Mean predicted risk',ylabel='Observed proportion');ax.legend(fontsize=8);ax.grid(alpha=.15)
  for ax in list(axes.flat)[len(keys):]:ax.set_visible(False)
  fig.savefig(ROOT/(setting+'_reliability.png'),dpi=160);fig.savefig(ROOT/(setting+'_reliability.pdf'));plt.close(fig)
 c.dump(ROOT/'execution_checks.json',dict(status='passed',calibrators=len(locked),previous_A_calibrators_reused=14,new_calibrators=len(locked)-14,base_refits=0,AP_AUROC_preserved=True,complete_separation_diagnostics_missing=int((r.diagnostic_status=='complete separation: no finite joint MLE').sum())))
if __name__=='__main__':main()
