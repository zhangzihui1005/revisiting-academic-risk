"""Core reviewer package: verify saved evidence or explicitly replay a key pipeline."""
import argparse, hashlib, importlib.util, json, shutil, subprocess, sys, zipfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
T='E4T_OULAD_temporal_generalization_v1.0'
U='E5A_UCI_information_horizons_v1.0'
F='EXPERIMENTS_FROZEN_v1.0'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def check():
    """Exact binary hashes; UTF-8 text may differ only by CRLF/LF line endings."""
    spec=load(HERE/'FILE_MANIFEST.json');failures=[];normalized=[]
    for n,item in spec['files'].items():
        p=HERE/n
        if not p.is_file():
            failures.append(n+': missing required file');continue
        data=p.read_bytes()
        if len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']:
            continue
        if item.get('comparison')=='utf8-lf':
            try:
                data.decode('utf-8')
                canonical=data.replace(b'\r\n',b'\n')
                if len(canonical)==item['lf_bytes'] and hashlib.sha256(canonical).hexdigest()==item['lf_sha256']:
                    normalized.append(n);continue
            except UnicodeDecodeError:
                pass
        failures.append(n+': SHA256/content mismatch')
    if failures:
        raise SystemExit('PACKAGE CHECK FAILED:\n'+'\n'.join(failures))
    print('PACKAGE VERIFIED:',len(spec['files'])+1,'required repository files;',len(normalized),'text files matched after CRLF/LF normalization',flush=True)
def restore_payload(out,prefix):
    import tarfile
    bundle=load(HERE/'evidence'/('bundle.json' if prefix=='core' else prefix+'_bundle.json'))
    idx=load(HERE/'evidence'/('SOURCE_INDEX.json' if prefix=='core' else prefix+'_SOURCE_INDEX.json'))
    if not out.exists():
        out.parent.mkdir(parents=True,exist_ok=True);archive=out.parent/('_'+prefix+'.tar.xz')
        with archive.open('wb') as f:
            for part in bundle['parts']:
                p=HERE/'evidence'/part['name'];assert sha(p)==part['sha256']
                with p.open('rb') as inp:shutil.copyfileobj(inp,f)
        assert sha(archive)==bundle['payload_sha256']
        out.mkdir()
        with tarfile.open(archive,'r:xz') as t:
            members=t.getmembers();assert {m.name for m in members}==set(idx)
            for m in members:
                assert m.isfile();p=(out/m.name).resolve();assert p.is_relative_to(out.resolve())
                p.parent.mkdir(parents=True,exist_ok=True)
                with t.extractfile(m) as inp,p.open('wb') as f:shutil.copyfileobj(inp,f)
        archive.unlink()
    for n,item in idx.items():
        assert (out/n).is_file() and sha(out/n)==item['sha256'],n
    print(prefix.upper()+' EVIDENCE VERIFIED:',len(idx),'unchanged source files',flush=True)
    return out
def restore(work):
    check()
    out=restore_payload(work/'frozen/outputs','core')
    restore_payload(work/'revision/research_project','revision')
    return out
def module(p):
    spec=importlib.util.spec_from_file_location('frozen_metric_definitions',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def verify(out,work):
    import numpy as np
    import pandas as pd
    frozen=module(out/F/'verify_results.py')
    gold=pd.read_csv(out/F/'all_recomputed_metrics.csv',float_precision='round_trip')
    rows=[];skipped=[];diffs=[];cache={}
    columns=['N','positive','prevalence','AP','nAP','AUROC','Brier','LogLoss','Lift_at_10']
    for _,r in gold.iterrows():
        rel=r.prediction_file.replace('\\','/');p=out/rel
        if not p.exists():skipped.append(dict(stage=r.stage,setting=r.setting,model=r.model));continue
        if rel not in cache:cache[rel]=pd.read_csv(p,float_precision='round_trip')
        d=cache[rel]
        if r.stage=='E3':y=d['y_'+r.setting.split('_',1)[1]];prob=d[r.model+'_probability'];keys=d.enrolment_key
        else:
            if r.stage=='E4T' and r.domain=='matched_modules':
                cfg=load(out/T/'config.json');d=d[d.module.isin(cfg[r.setting]['matched_modules'])]
            y=d.y;prob=d[r.model] if r.stage=='E6C' else d.probability;keys=d.key
        m=frozen.metric(y,prob,keys)
        for c in columns:
            assert np.isclose(m[c],r[c],atol=1e-7,rtol=0,equal_nan=True),(r.stage,r.setting,r.model,c,m[c],r[c])
            if np.isfinite(m[c]) and np.isfinite(r[c]):diffs.append(abs(float(m[c])-float(r[c])))
        rows.append(dict(stage=r.stage,setting=r.setting,domain=r.domain,model=r.model,cohort=r.cohort,**m))
    assert rows,'No prediction metrics checked'
    intervals=pd.read_csv(out/F/'Results_Table2_contrasts.csv',float_precision='round_trip');ic=[]
    for _,r in intervals.iterrows():
        d=pd.read_csv(out/r.bootstrap.replace('\\','/'),float_precision='round_trip').delta_nAP.dropna()
        qs=d.quantile([.025,.975,.025/r.family_m,1-.025/r.family_m]).to_numpy()
        assert len(d)==r.frozen_replicates
        assert np.allclose(qs,[r.CI95_low,r.CI95_high,r.adjusted_CI_low,r.adjusted_CI_high],atol=1e-10,rtol=0)
        ic.append(dict(contrast=r.contrast,existing_draws=len(d),status='passed'))
    cal=[]
    for name in ['B_Day28_failure','B_Day28_withdrawal','B_E4T-B_day56_failure_secondary']:
        base=out/'E6C_probability_transportability_v1.0';d=pd.read_csv(base/name/'target_predictions.csv',float_precision='round_trip')
        raw=frozen.metric(d.y,d.raw,d.key);adj=frozen.metric(d.y,d.calibrated,d.key)
        assert abs(raw['AP']-adj['AP'])<1e-12 and abs(raw['AUROC']-adj['AUROC'])<1e-12
        goldcal=pd.read_csv(base/'paired_calibration_changes.csv',float_precision='round_trip');r=goldcal[goldcal.experiment.eq(name)].iloc[0]
        reps=pd.read_csv(base/name/'paired_bootstrap.csv',float_precision='round_trip')
        for c in ['Brier','LogLoss']:
            lo,hi=reps['delta_'+c].quantile([.025,.975]);assert np.allclose([adj[c]-raw[c],lo,hi],[r['delta_'+c],r['delta_'+c+'_CI_low'],r['delta_'+c+'_CI_high']],atol=1e-10,rtol=0)
        cal.append(dict(setting=name,delta_AP=adj['AP']-raw['AP'],delta_AUROC=adj['AUROC']-raw['AUROC'],delta_Brier=adj['Brier']-raw['Brier'],delta_LogLoss=adj['LogLoss']-raw['LogLoss']))
    pd.DataFrame(rows).to_csv(work/'verified_core_metrics.csv',index=False)
    pd.DataFrame(ic).to_csv(work/'verified_existing_intervals.csv',index=False)
    pd.DataFrame(cal).to_csv(work/'verified_calibration.csv',index=False)
    report=dict(status='passed',prediction_metric_rows=len(rows),metric_comparisons=len(rows)*len(columns),max_abs_difference=max(diffs),
        frozen_summary_rows_without_bundled_predictions=len(skipped),existing_interval_rows=len(ic),calibration_settings=len(cal),
        fits=0,new_bootstrap_draws=0,scope='Core subset only. Other E1–E9 tables are frozen summaries, not newly recomputed here.')
    dump(work/'VERIFICATION.json',report);print(json.dumps(report),flush=True)
def run(p,*args):subprocess.run([sys.executable,'-u',str(p),*map(str,args)],check=True)
def prepare_uci(out,dest):
    assert not dest.exists(),'Use a fresh work directory.'
    src=out/U;dest.mkdir(parents=True)
    names=set(load(src/'run_manifest.json')['hashes'])|{'run_manifest.json','run.py','audit.py','analyze.py','report.py','PROTOCOL.txt','requirements-lock.txt'}
    for n in sorted(names):
        p=dest/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/n,p)
    run(dest/'run.py','preaudit')
    return dest
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('action',choices=['check','restore','verify','preflight','raw-oulad','rerun-uci'])
    a.add_argument('--workdir',type=Path,default=HERE/'_work');a.add_argument('--oulad',type=Path);a.add_argument('--workers',type=int,default=3)
    v=a.parse_args();work=v.workdir.resolve()
    if v.action=='check':check();return
    if v.action=='raw-oulad':
        if not v.oulad or not v.oulad.is_file():raise SystemExit('Supply --oulad original ZIP; see docs/DATA_SOURCES.txt.')
        if sha(v.oulad)!='f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4':raise SystemExit('Raw OULAD SHA256 does not match the frozen study.')
    out=restore(work)
    if v.action=='restore':return
    if v.action=='verify':
        verify(out,work)
        import revision
        revision.verify(work/'revision/research_project',work)
        return
    if v.action in ['preflight','rerun-uci']:
        dest=prepare_uci(out,work/('uci_input_check' if v.action=='preflight' else 'uci_replay'))
        if v.action=='preflight':
            for h in [28,56]:assert (out/T/f'frozen_day{h}_features.csv').is_file()
            for n in ['run.py','e1_reference.py']:assert (out/F/'frozen_code'/n).is_file()
            dump(work/'PREFLIGHT.json',dict(status='passed',uci_manifest_hashes_and_splits=True,raw_oulad_reference_files=True,fits=0))
            import revision
            dump(work/'REVISION_PREFLIGHT.json',dict(status='passed',jobs=revision.preflight(work/'revision/research_project'),fits=0))
            print('PREFLIGHT PASSED; no fitting performed');return
        run(dest/'run.py','train','--workers',v.workers);run(dest/'run.py','score')
        for script in ['audit.py','analyze.py','report.py']:run(dest/script)
        import pandas as pd,numpy as np
        comparisons=[]
        for exp in load(out/U/'config.json'):
            for fam in ['LR','XGBoost','CatBoost']:
                old=out/U/exp/fam;new=dest/exp/fam
                x=load(old/'selection.json');y=load(new/'selection.json');assert x['params']==y['params'] and x['best_trial']==y['best_trial']
                p=pd.read_csv(old/'test_predictions.csv',float_precision='round_trip');q=pd.read_csv(new/'test_predictions.csv',float_precision='round_trip')
                assert p.key.equals(q.key);delta=float(np.max(abs(p.probability-q.probability)));assert delta<1e-7
                comparisons.append(dict(experiment=exp,model=fam,max_prediction_difference=delta))
        dump(work/'UCI_REPLAY_COMPARISON.json',dict(status='passed',comparisons=comparisons));return
    run(out/F/'reproduce_raw.py','--raw',v.oulad.resolve(),'--destination',work/'raw_oulad_replay','--reference',out/T)
if __name__=='__main__':main()
