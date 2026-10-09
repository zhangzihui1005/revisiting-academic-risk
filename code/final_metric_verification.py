"""Read-only numerical closeout of frozen E1-E9; write only this closeout directory."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
HERE=Path(__file__).resolve().parent;OUT=HERE.parent
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
INPUTS={}
def read(p):INPUTS[str(p.relative_to(OUT)).replace('\\','/')]=sha(p);return pd.read_csv(p,float_precision='round_trip')
def metric(y,p,key):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pi=y.mean();ap=average_precision_score(y,p);k=int(np.ceil(.1*len(y)));order=np.lexsort((np.asarray(key,dtype=str),-p));return dict(N=len(y),positive=int(y.sum()),prevalence=pi,AP=ap,nAP=(ap-pi)/(1-pi),AUROC=roc_auc_score(y,p),Brier=brier_score_loss(y,p),LogLoss=log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1]),Lift_at_10=y[order[:k]].mean()/pi)
def snapshot():
 p=HERE/'historical_hashes.json'
 if not p.exists():dump(p,{str(f.relative_to(OUT)).replace('\\','/'):sha(f) for d in OUT.iterdir() if d.is_dir() and d!=HERE for f in d.rglob('*') if f.is_file() and '__pycache__' not in f.parts})
def verify_metrics():
 rows=[];diffs=[]
 def add(stage,setting,domain,model,selected,pred,y,p,key,old=None,cohort='dynamic'):
  m=metric(y,p,key);r=dict(stage=stage,setting=setting,domain=domain,model=model,selected=bool(selected),cohort=cohort,prediction_file=str(pred.relative_to(OUT)).replace('\\','/'),**m);rows.append(r)
  if old is not None:
   for col in ['AP','nAP','AUROC','Brier','Lift_at_10','LogLoss']:
    if col in old and pd.notna(old[col]):
     delta=m[col]-old[col];diffs.append(dict(stage=stage,setting=setting,domain=domain,model=model,cohort=cohort,metric=col,recomputed=m[col],archived=old[col],difference=delta))
 root=OUT/'E3_oulad_multihorizon_v1.0'
 for cohort,file in [('dynamic','all_models_all_horizons.csv'),('common','stable_all_models_trajectory.csv')]:
  for _,r in read(root/file).iterrows():
   h=int(r.landmark);p=root/f'day{h}'/(r.task+('_stable' if cohort=='common' else '')+'_test_predictions.csv');d=read(p);scores=d[r.model+'_probability'];add('E3',f'Day{h}_{r.task}','internal',r.model,r.selected_on_validation,p,d['y_'+r.task],scores,d.enrolment_key,r,cohort)
 for stage,name in [('E4A','E4A_day28_student_disjoint_LOMO_v1.0'),('E4B','E4B_day56_LOMO_secondary_v1.0')]:
  root=OUT/name
  for _,r in read(root/'all_models_test_results.csv').iterrows():
   p=root/'folds'/r.heldout_module/r.task/r.model/'test_predictions.csv';d=read(p);scores=d.probability.astype(np.float32).astype(float) if r.model=='XGBoost' else d.probability;add(stage,r.task,r.heldout_module,r.model,r.is_validation_selected,p,d.y,scores,d.enrolment_key,r)
 for stage,name in [('E4T','E4T_OULAD_temporal_generalization_v1.0'),('E5A','E5A_UCI_information_horizons_v1.0'),('E5B','E5B_UCI_LOCO_v1.0')]:
  root=OUT/name;cfg=json.loads((root/'config.json').read_text())
  for _,r in read(root/'all_models_test_results.csv').iterrows():
   p=root/r.experiment/r.model/'test_predictions.csv';d=read(p)
   if r.scope=='matched_modules':d=d[d.module.isin(cfg[r.experiment]['matched_modules'])]
   add(stage,r.experiment,r.scope,r.model,r.selected,p,d.y,d.probability,d.key,r,cohort='retrospective information set' if stage.startswith('E5') else 'dynamic')
 root=OUT/'E6C_probability_transportability_v1.0';old=read(root/'target_calibration_metrics.csv');paired=read(root/'paired_calibration_changes.csv');cal=[]
 for exp,g in old.groupby('experiment'):
  p=root/exp/'target_predictions.csv';d=read(p);mm={}
  for method in ['raw','calibrated','source_prior']:
   row=g[g.method.eq(method)].iloc[0];mm[method]=metric(d.y,d[method],d.key);add('E6C',row.setting,exp,method,True,p,d.y,d[method],d.key,row)
  q=paired[paired.experiment.eq(exp)].iloc[0];assert abs(mm['raw']['AP']-mm['calibrated']['AP'])<1e-12 and abs(mm['raw']['AUROC']-mm['calibrated']['AUROC'])<1e-12
  z=dict(experiment=exp,delta_AP=mm['calibrated']['AP']-mm['raw']['AP'],delta_AUROC=mm['calibrated']['AUROC']-mm['raw']['AUROC'])
  reps=read(root/exp/'paired_bootstrap.csv')
  for name in ['Brier','LogLoss']:
   delta=mm['calibrated'][name]-mm['raw'][name];lo,hi=reps['delta_'+name].quantile([.025,.975]);assert np.allclose([delta,lo,hi],[q['delta_'+name],q['delta_'+name+'_CI_low'],q['delta_'+name+'_CI_high']],atol=1e-10);z['delta_'+name]=delta
  cal.append(z)
 root=OUT/'E6A_simple_ensemble_gate_v1.0';old=read(root/'target_domain_metrics.csv')
 for exp,g in old.groupby('experiment'):
  p=root/exp/'target_predictions.csv';d=read(p)
  for _,r in g.iterrows():
   domain=d.module.astype(str) if r.environment=='B' else d.domain.astype(str);z=d[domain.eq(str(r.domain))];add('E6A',r.setting,str(r.domain),r.method,False,p,z.y,z[r.method],z.key,r)
 root=OUT/'E8_OULAD_grouped_ablation_v1.0'
 for _,r in read(root/'target_metrics.csv').iterrows():
  p=(OUT/'E4T_OULAD_temporal_generalization_v1.0'/r.experiment/r.family/'test_predictions.csv') if r.group=='Full' else root/r.experiment/'target_predictions.csv';d=read(p);add('E8',r.experiment,r.group,r.family,False,p,d.y,d.probability,d.key,r)
 result=pd.DataFrame(rows);dd=pd.DataFrame(diffs);result.to_csv(HERE/'all_recomputed_metrics.csv',index=False);dd.to_csv(HERE/'metric_reconciliation.csv',index=False);pd.DataFrame(cal).to_csv(HERE/'calibration_reconciliation.csv',index=False)
 # Preserve discrepancies first; do not silently pass or overwrite old tables.
 bad=dd[dd.difference.abs()>1e-7];bad.to_csv(HERE/'metric_discrepancies.csv',index=False)
 selected=result[result.selected&result.stage.ne('E6C')].copy();summary=[]
 for (stage,setting,cohort),z in selected.groupby(['stage','setting','cohort']):
  domain_equal=stage in ['E4A','E4B'];row=dict(stage=stage,setting=setting,cohort=cohort,aggregation='equal held-out modules' if domain_equal else 'pooled rows within listed scope',units=len(z),models=';'.join(sorted(z.model.unique())))
  if stage=='E4T':
   row['units']=1
   for _,v in z.iterrows():summary.append(dict(**row,scope=v.domain,**{m:v[m] for m in ['N','AP','nAP','AUROC','Brier','LogLoss','Lift_at_10']}))
  elif stage!='E5B':summary.append(dict(**row,scope='all',**{m:z[m].sum() if m=='N' else z[m].mean() for m in ['N','AP','nAP','AUROC','Brier','LogLoss','Lift_at_10']}))
 z=selected[selected.stage.eq('E5B')];summary.append(dict(stage='E5B',setting='UCI_H1_LOCO',cohort='retrospective information set',aggregation='equal 17 held-out courses',units=len(z),models='source-selected',scope='all',**{m:z[m].sum() if m=='N' else z[m].mean() for m in ['N','AP','nAP','AUROC','Brier','LogLoss','Lift_at_10']}));pd.DataFrame(summary).to_csv(HERE/'Results_Table1_selected_performance.csv',index=False)
 dump(HERE/'metric_verification.json',dict(status='passed' if len(bad)==0 else 'review_required',metric_rows=len(result),archived_comparisons=len(dd),max_abs_difference=float(dd.difference.abs().max()),discrepancies=len(bad),calibration_checks=len(cal),logloss_clip=1e-12,tie_rule='ascending record key after descending probability'))
 print('METRIC AUDIT',len(result),'rows; discrepancies',len(bad),flush=True)
def contrasts():
 rows=[]
 def add(label,priority,file,rep,point,low,high,family,m):
  bs=read(rep).delta_nAP.dropna();lo,hi=bs.quantile([.025,.975]);assert np.allclose([lo,hi],[low,high],atol=1e-10);alo,ahi=bs.quantile([.025/m,1-.025/m]);rows.append(dict(contrast=label,priority=priority,delta_nAP=point,CI95_low=lo,CI95_high=hi,family=family,family_m=m,Bonferroni_marginal_coverage=1-.05/m,adjusted_CI_low=alo,adjusted_CI_high=ahi,frozen_replicates=len(bs),source=str(file.relative_to(OUT)),bootstrap=str(rep.relative_to(OUT))))
 root=OUT/'E3_oulad_multihorizon_v1.0';p=root/'stable_paired_horizon_gains.csv';d=read(p)
 for task in ['failure','withdrawal']:
  r=d[d.task.eq(task)&d.from_day.eq(28)&d.to_day.eq(56)].iloc[0];add('Internal '+task+' Day28→56','primary' if task=='failure' else 'primary outcome control',p,root/f'stable_bootstrap_{task}_28_56.csv',r.delta_nAP,r.nAP_CI_low,r.nAP_CI_high,'manuscript_information_contrasts',5)
 root=OUT/'E4T_OULAD_temporal_generalization_v1.0';p=root/'paired_horizon_comparison.csv';d=read(p);r=d[d.scope.eq('all')&d.comparison.eq('Validation-selected')].iloc[0];add('Future-year Failure Day28→56','primary',p,root/'paired/all/Validation-selected/paired_bootstrap.csv',r.delta_nAP,r.delta_nAP_CI_low,r.delta_nAP_CI_high,'manuscript_information_contrasts',5)
 root=OUT/'E5A_UCI_information_horizons_v1.0';p=root/'paired_horizon_comparison.csv';d=read(p)
 for earlier,later,priority in [('H0_strict','H1','primary'),('H1','H2','secondary')]:
  r=d[d.earlier.eq(earlier)&d.later.eq(later)&d.comparison.eq('Validation-selected')].iloc[0];add('UCI '+earlier+'→'+later,priority,p,root/'paired'/(earlier+'_to_'+later)/'Validation-selected/paired_bootstrap.csv',r.delta_nAP,r.delta_nAP_CI_low,r.delta_nAP_CI_high,'manuscript_information_contrasts',5)
 root=OUT/'E4B_day56_LOMO_secondary_v1.0';p=root/'common_cohort_comparison.csv';d=read(p)
 for _,r in d[d.task.eq('failure')&d.comparison.eq('Validation-selected')].iterrows():add('LOMO Failure '+r.heldout_module+' Day28→56','secondary domain diagnostic',p,root/'common_cohort'/r.heldout_module/'failure/paired_bootstrap.csv',r.delta_nAP,r.delta_nAP_CI_low,r.delta_nAP_CI_high,'LOMO_seven_domains',7)
 pd.DataFrame(rows).to_csv(HERE/'Results_Table2_contrasts.csv',index=False)
 registry=[('RQ1','primary','Failure Day28→56: internal, LOMO, future-year; UCI H0→H1','LOMO macro descriptive; inferential contrasts use fixed cohorts'),('RQ1','primary control','Withdrawal internal common-cohort Day28→56','No equivalence claim when CI crosses zero'),('RQ1','secondary','UCI H1→H2; matched-module and dynamic-cohort sensitivity','Different estimands explicitly labeled'),('RQ2','primary descriptive','Source-selected vs fixed families; E6A source gate','NO GO unchanged; target not used to trigger optimizer'),('RQ3','secondary','Source-only calibration transport','Domain-specific paired intervals exploratory, no universal benefit claim'),('RQ3','post-hoc/explanatory','E7 coefficients/SHAP, E8 fixed-parameter group removal','No causality; no new models'),('RQ3','post-hoc sensitivity','UCI calibration equal-domain/exclude tiny course/record weighting','Keep all versions; no result-based preference')]
 pd.DataFrame(registry,columns=['RQ','reporting_priority','contrast_or_analysis','constraint']).to_csv(HERE/'claim_registry.csv',index=False)
def interpretation():
 root=OUT/'E7_interpretation_v1.0';coef=read(root/'failure_coefficient_evolution.csv');shap=read(root/'failure_SHAP_evolution.csv');z=coef.merge(shap,on='feature');out=[]
 groups=json.loads((OUT/'E8_OULAD_grouped_ablation_v1.0/feature_groups.json').read_text());featuregroup={f:g for g,fs in groups.items() for f in fs}
 for _,r in z.iterrows():
  for h in [28,56]:
   b=r[f'beta_standardized_{h}'];rho=r[f'value_SHAP_spearman_{h}'];out.append(dict(feature=r.feature,group=featuregroup[r.feature],horizon=h,LR_beta=b,CatBoost_value_SHAP_spearman=rho,direction_agrees=bool(np.sign(b)==np.sign(rho)),interpretation='descriptive sign agreement only' if np.sign(b)==np.sign(rho) else 'discordant marginal directions: retain nonlinear/correlated-feature caveat'))
 for r in out:r['task']='failure'
 wc=read(root/'OULAD_standardized_coefficients.csv');ws=read(root/'feature_SHAP_summary.csv');wc=wc[wc.model.eq('Day28_withdrawal')];ws=ws[ws.model.eq('OULAD_Day28_withdrawal')]
 for _,r in wc.merge(ws,on='feature').iterrows():
  b=r.beta_standardized;rho=r.value_SHAP_spearman;out.append(dict(feature=r.feature,group=featuregroup[r.feature],horizon=28,task='withdrawal',LR_beta=b,CatBoost_value_SHAP_spearman=rho,direction_agrees=bool(np.sign(b)==np.sign(rho)),interpretation='descriptive sign agreement only' if np.sign(b)==np.sign(rho) else 'discordant marginal directions: retain nonlinear/correlated-feature caveat'))
 pd.DataFrame(out).to_csv(HERE/'interpretation_direction_audit.csv',index=False)
 fixed=read(OUT/'E8_OULAD_grouped_ablation_v1.0/fixed_horizon_ablation.csv');gain=read(OUT/'E8_OULAD_grouped_ablation_v1.0/information_gain_ablation.csv');allshap=read(root/'feature_SHAP_summary.csv');grows=[]
 for group,fs in groups.items():
  for h in [28,56]:
   s=allshap[allshap.model.eq('OULAD_Day28_failure' if h==28 else 'OULAD_E4T-B_day56_failure_secondary')&allshap.feature.isin(fs)];bb=coef[coef.feature.isin(fs)];row=dict(group=group,horizon=h,sum_abs_LR_beta=bb[f'abs_beta_{h}'].sum(),sum_feature_mean_abs_CatBoost_SHAP=s.mean_abs_SHAP.sum())
   for fam in ['LR','CatBoost']:
    q=fixed[fixed.family.eq(fam)&fixed.horizon.eq(h)&fixed.removed_group.eq(group)].iloc[0];row[f'{fam}_removal_delta_nAP']=q.delta_nAP
   grows.append(row)
 pd.DataFrame(grows).to_csv(HERE/'Results_Table3_group_consistency.csv',index=False)
 aliases={'prior_background':'prior history','participation':'engagement','academic_progress':'progress','academic_performance':'performance','administrative_context':'administrative context'};mapping=[]
 # Preserve the exact E7 mapping; the four headline concepts do not absorb context.
 for (dataset,feature),g in allshap.groupby(['dataset','feature']):
  assert g.concept.nunique()==1;concept=g.concept.iloc[0];mapping.append(dict(dataset=dataset,feature=feature,E7_concept=concept,manuscript_concept=aliases[concept],measured=True,scope='OULAD frozen 16 features' if dataset=='OULAD' else 'UCI H1 27 features',limitation='concept-level replication; no feature transfer'))
 for concept in ['academic_progress','academic_performance']:mapping.append(dict(dataset='OULAD',feature='NOT MEASURED',E7_concept=concept,manuscript_concept=aliases[concept],measured=False,scope='no assessment/grade features in frozen OULAD inputs',limitation='missing, not zero importance'))
 pd.DataFrame(mapping).to_csv(HERE/'Results_Table4_frozen_concept_mapping.csv',index=False)
 dump(HERE/'interpretation_verification.json',dict(status='passed_with_disclosed_discordance',feature_horizon_pairs=len(out),direction_disagreements=sum(not r['direction_agrees'] for r in out),mapping_rows=len(mapping),no_new_SHAP_models=True,claims='F4 dependence exceeds F3 for gain; F1/F2 gain increase is not absolute performance improvement'))
def main():
 snapshot();verify_metrics();contrasts();interpretation();dump(HERE/'checked_input_hashes.json',INPUTS);print('CLOSEOUT TABLES BUILT',flush=True)
if __name__=='__main__':main()
