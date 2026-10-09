"""Independent schema audit; no UCI model training in E1."""
import argparse,json,zipfile
from pathlib import Path
import pandas as pd

def check_uci(path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        d=pd.read_csv(z.open('data.csv'),sep=';',na_values=['?'])
    d.columns=d.columns.str.strip()
    sem1=[c for c in d if c.startswith('Curricular units 1st sem')]
    sem2=[c for c in d if c.startswith('Curricular units 2nd sem')]
    h0=[c for c in d if c not in sem1+sem2+['Target']]
    assert (len(h0),len(sem1),len(sem2))==(24,6,6)
    assert len(d)==4424 and d.Course.nunique()==17
    horizons={'H0':h0,'H1':h0+sem1,'H2':h0+sem1+sem2}
    (out/'uci_horizons.json').write_text(json.dumps(horizons,ensure_ascii=False,indent=2),encoding='utf-8')
    audit={'records':len(d),'features':36,'courses':d.Course.nunique(),'label_counts':d.Target.value_counts().to_dict(),
        'missing_cells':int(d.isna().sum().sum()),'duplicate_rows':int(d.duplicated().sum()),
        'horizon_counts':{k:len(v) for k,v in horizons.items()},
        'caution':'H0 is a schema grouping, not a verified availability guarantee. Debtor, Tuition fees up to date, Scholarship holder and macroeconomic fields need acquisition-time verification; no student year or dropout date is available. H1/H2 cannot guarantee active-at-horizon cohorts.'}
    (out/'uci_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('UCI_AUDIT',json.dumps(audit),flush=True)
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--uci',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();check_uci(a.uci,a.output)
