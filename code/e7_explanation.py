import sys,os
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from scipy.stats import spearmanr
from catboost import Pool
from xgboost import DMatrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import e6_common as c
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;T=OUT/'E4T_OULAD_temporal_generalization_v1.0';U=OUT/'E5A_UCI_information_horizons_v1.0'
def savefig(fig,n):fig.savefig(ROOT/(n+'.png'),dpi=170,bbox_inches='tight');fig.savefig(ROOT/(n+'.pdf'),bbox_inches='tight');plt.close(fig)
def choose(d,keys=None):
 d=d.sort_values('key');rng=np.random.default_rng(c.SEED)
 if keys is None:keys=d.key.to_numpy()[np.sort(rng.choice(len(d),min(1000,len(d)),replace=False))]
 return d.set_index('key',drop=False).loc[keys].reset_index(drop=True)
def concept(feature,kind):
 if kind=='OULAD':return 'prior_background' if feature=='num_of_prev_attempts' else 'administrative_context' if feature in ['studied_credits','registration_lead_time'] else 'participation'
 if feature in ['Previous qualification','Previous qualification (grade)','Admission grade']:return 'prior_background'
 if feature in ['Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (without evaluations)']:return 'participation'
 if feature in ['Curricular units 1st sem (approved)','Curricular units 1st sem (credited)']:return 'academic_progress'
 if feature=='Curricular units 1st sem (grade)':return 'academic_performance'
 return 'administrative_context'
def attribution(label,kind,z,features,values,bias,raw,p,catfeatures=[]):
 dest=ROOT/label;dest.mkdir();err=np.max(np.abs(values.sum(axis=1)+bias-raw));assert err<(2e-5 if kind=='UCI' else 1e-7),err;assert np.max(abs(expit(raw)-p))<1e-6
 z[['key','row_id','id','y']+features].to_csv(dest/'explanation_sample.csv',index=False);pd.DataFrame(values,columns=features).assign(key=z.key.to_numpy(),expected_value=bias,raw_margin=raw).to_csv(dest/'shap_values.csv.gz',index=False)
 rows=[];cr=[]
 for i,f in enumerate(features):
  rho=np.nan if f in catfeatures or z[f].nunique()<2 or np.std(values[:,i])<1e-15 else spearmanr(z[f],values[:,i]).statistic
  rows.append(dict(model=label,dataset=kind,feature=f,concept=concept(f,kind),mean_abs_SHAP=np.mean(abs(values[:,i])),mean_SHAP=np.mean(values[:,i]),value_SHAP_spearman=rho))
 for g in ['prior_background','participation','academic_progress','academic_performance','administrative_context']:
  ix=[i for i,f in enumerate(features) if concept(f,kind)==g];v=values[:,ix].sum(axis=1) if ix else None;cr.append(dict(model=label,dataset=kind,concept=g,measured=bool(ix),feature_count=len(ix),mean_abs_group_SHAP=np.mean(abs(v)) if ix else np.nan))
 total=sum(r['mean_abs_group_SHAP'] for r in cr if r['measured'])
 for r in cr:r['normalized_group_share']=r['mean_abs_group_SHAP']/total if r['measured'] else np.nan
 return rows,cr,dict(model=label,N=len(z),max_additivity_error=err,probability_link_error=np.max(abs(expit(raw)-p)))
def main():
 cfg=c.load(T/'config.json');coefs=[];attrib=[];concepts=[];checks=[];frames={};hashes={}
 for ex,v in cfg.items():
  d=pd.read_csv(T/v['data']);tr=pd.read_csv(T/ex/'train.csv');frames[ex]=d[d.row_id.isin(tr.row_id)].copy();m=joblib.load(T/ex/'LR/model.joblib');sc=m.steps[0][1];lr=m.steps[-1][1]
  for i,f in enumerate(v['features']):coefs.append(dict(model=ex,feature=f,beta_standardized=lr.coef_[0,i],abs_beta=abs(lr.coef_[0,i]),source_mean=sc.mean_[i],source_SD=sc.scale_[i],beta_raw_unit=lr.coef_[0,i]/sc.scale_[i],concept=concept(f,'OULAD')))
  for family in ['LR','CatBoost']:hashes[str((T/ex/family/'model.joblib').relative_to(OUT))]=c.sha(T/ex/family/'model.joblib')
 pd.DataFrame(coefs).to_csv(ROOT/'OULAD_standardized_coefficients.csv',index=False)
 early='Day28_failure';late='E4T-B_day56_failure_secondary';common=sorted(set(frames[early].key)&set(frames[late].key));sample=choose(frames[early][frames[early].key.isin(common)]);keys=sample.key.to_numpy()
 for ex,v in cfg.items():
  z=choose(frames[ex],keys if ex in [early,late] else None);m=joblib.load(T/ex/'CatBoost/model.joblib');pool=Pool(z[v['features']]);sv=m.get_feature_importance(pool,type='ShapValues',thread_count=6);raw=m.predict(pool,prediction_type='RawFormulaVal');p=m.predict_proba(pool)[:,1];rr,cr,ck=attribution('OULAD_'+ex,'OULAD',z,v['features'],sv[:,:-1],sv[:,-1],raw,p);attrib+=rr;concepts+=cr;checks.append(ck);print('SHAP',ex,flush=True)
 v=c.load(U/'config.json')['H1'];d=pd.read_csv(U/v['data']);tr=pd.read_csv(U/'H1/train.csv');z=choose(d[d.row_id.isin(tr.row_id)]);choice=pd.read_csv(U/'source_selected_models.csv').set_index('experiment').loc['H1','selected_model'];assert choice=='XGBoost';m=joblib.load(U/'H1'/choice/'model.joblib');pre=m.steps[0][1];tree=m.steps[-1][1];x=pre.transform(z[v['features']]);booster=tree.get_booster();dm=DMatrix(x);sv=booster.predict(dm,pred_contribs=True);raw=booster.predict(dm,output_margin=True);p=m.predict_proba(z[v['features']])[:,1]
 cats=v['categorical'];nums=[f for f in v['features'] if f not in cats];mapping=[f for f,levels in zip(cats,pre.named_transformers_['categorical'].categories_) for _ in levels]+nums;assert len(mapping)==x.shape[1];grouped=np.column_stack([sv[:,[i for i,n in enumerate(mapping) if n==f]].sum(axis=1) for f in v['features']]);rr,cr,ck=attribution('UCI_H1_XGBoost','UCI',z,v['features'],grouped,sv[:,-1],raw,p,cats);attrib+=rr;concepts+=cr;checks.append(ck);hashes[str((U/'H1'/choice/'model.joblib').relative_to(OUT))]=c.sha(U/'H1'/choice/'model.joblib')
 ar=pd.DataFrame(attrib);ar.to_csv(ROOT/'feature_SHAP_summary.csv',index=False);pd.DataFrame(concepts).to_csv(ROOT/'concept_attribution_summary.csv',index=False);pd.DataFrame(checks).to_csv(ROOT/'SHAP_additivity_checks.csv',index=False)
 coef=pd.DataFrame(coefs);delta=coef[coef.model.eq(early)].merge(coef[coef.model.eq(late)],on='feature',suffixes=('_28','_56'));delta['delta_abs_beta']=delta.abs_beta_56-delta.abs_beta_28;delta['delta_beta']=delta.beta_standardized_56-delta.beta_standardized_28;delta.to_csv(ROOT/'failure_coefficient_evolution.csv',index=False)
 sd=ar[ar.model.eq('OULAD_'+early)].merge(ar[ar.model.eq('OULAD_'+late)],on='feature',suffixes=('_28','_56'));sd['delta_mean_abs_SHAP']=sd.mean_abs_SHAP_56-sd.mean_abs_SHAP_28;sd.to_csv(ROOT/'failure_SHAP_evolution.csv',index=False)
 stab=[]
 for dirname,task,h in [('E4A_day28_student_disjoint_LOMO_v1.0','withdrawal',28),('E4A_day28_student_disjoint_LOMO_v1.0','failure',28),('E4B_day56_LOMO_secondary_v1.0','failure',56)]:
  vals=[]
  for g in [s*3 for s in 'ABCDEFG']:
   path=OUT/dirname/'folds'/g/task/'LR/model.joblib';obj=joblib.load(path);mm=obj['model'];vals.append(mm.steps[-1][1].coef_[0])
  vals=np.vstack(vals)
  for i,f in enumerate(cfg[early]['features']):stab.append(dict(task=task,horizon=h,feature=f,mean_beta=vals[:,i].mean(),SD_beta=vals[:,i].std(),positive_folds=int((vals[:,i]>0).sum()),negative_folds=int((vals[:,i]<0).sum())))
 pd.DataFrame(stab).to_csv(ROOT/'LOMO_coefficient_sign_stability.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(15,6.2),layout='constrained')
 for ax,ex in zip(axes,cfg):
  q=coef[coef.model==ex].sort_values('abs_beta');ax.barh(q.feature,q.beta_standardized,color=['#bd6741' if x>0 else '#327da5' for x in q.beta_standardized]);ax.axvline(0,color='gray',lw=.7);ax.set(title=ex.replace('E4T-B_day56_failure_secondary','Day56_failure'),xlabel='Standardized beta (source SD)');ax.tick_params(axis='y',labelsize=8)
 savefig(fig,'OULAD_standardized_beta')
 fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained');q=delta.sort_values('delta_abs_beta');axes[0].barh(q.feature,q.delta_abs_beta,color='#297a9e');axes[0].set(title='LR: change in absolute standardized beta',xlabel='Day56 − Day28');q=sd.sort_values('delta_mean_abs_SHAP');axes[1].barh(q.feature,q.delta_mean_abs_SHAP,color='#b66b41');axes[1].set(title='CatBoost: change in mean |TreeSHAP|',xlabel='Day56 − Day28 (raw-margin units)')
 for ax in axes:ax.axvline(0,color='gray',lw=.8);ax.tick_params(axis='y',labelsize=8)
 savefig(fig,'failure_feature_evolution')
 cc=pd.DataFrame(concepts);labels=['OULAD_Day28_failure','OULAD_'+late,'UCI_H1_XGBoost'];fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');pivot=cc[cc.model.isin(labels)].pivot(index='concept',columns='model',values='normalized_group_share').reindex(columns=labels);pivot.columns=['OULAD Failure Day28','OULAD Failure Day56','UCI H1'];pivot.plot.bar(ax=ax);ax.set(ylabel='Normalized mean |group contribution|',xlabel='',title='Concept-level attribution (source samples)');ax.tick_params(axis='x',rotation=15);ax.legend(fontsize=8);fig.text(.01,-.10,'OULAD academic progress/performance are not measured: missing bars do not mean zero importance.',fontsize=9);savefig(fig,'concept_replication')
 c.dump(ROOT/'input_model_hashes.json',hashes)
 for p,h in hashes.items():assert c.sha(OUT/p)==h
 c.dump(ROOT/'execution_checks.json',dict(status='passed',primary_LR_models=3,nonlinear_models=4,SHAP_samples=len(checks),common_failure_source_sample_keys=len(keys),new_models_fitted=0,additivity_verified=True,groupings_frozen_before_attributions=True,warning='correlated features, changing scales and different learners prevent causal interpretation'))
if __name__=='__main__':main()
