"""Step 12. Collect the main numbers into results/results.json."""
import pickle, json
import numpy as np, pandas as pd
from config import CACHE, RESULTS
from core import wilson
J={}
J['standards']=pd.read_csv(RESULTS / 'out_table_standards.csv').round(4).to_dict('records')
S=pd.read_csv(RESULTS / 'out_shortwindow.csv'); J['shortwindow']=S.round(4).to_dict('records')
res=pickle.load(open(CACHE / 'cv.pkl','rb')); R=res[(120,'drowsy')]
cv=R[(R['mode']=='cv5')&(R.acap==0.1)]; s0=cv[cv.seed==0]; lo=R[(R['mode']=='loso')&(R.acap==0.1)]
rows=[]
for fam in ['stea','stea_only','relax_p1','relax_multi','relax_p30','relax_ecd','count30','cusum','ewma']:
    for b in [2.5,5.0,8.0]:
        g=cv[(cv.fam==fam)&(cv.b==b)]; r0=s0[(s0.fam==fam)&(s0.b==b)].iloc[0]; rl=lo[(lo.fam==fam)&(lo.b==b)].iloc[0]
        ci=wilson(r0.cov_n,r0.cov_d)
        rows.append(dict(fam=fam,b=b,cov_n0=int(r0.cov_n),cov_d=int(r0.cov_d),ci_lo=ci[0],ci_hi=ci[1],far0=r0.far,act0=r0.act,lead0=r0.lead_med,
            cov_mean=g['cov'].mean(),cov_sd=g['cov'].std(),far_mean=g.far.mean(),far_sd=g.far.std(),act_mean=g.act.mean(),loso_n=int(rl.cov_n),loso_far=rl.far,loso_act=rl.act))
J['cv']=rows
for k,f in [('acap',RESULTS / 'out_acap.csv'),('lambda',RESULTS / 'out_lambda_profile.csv'),('stats',RESULTS / 'out_stats.csv'),('sens',RESULTS / 'out_sensitivity.csv'),('base_sens',RESULTS / 'out_base_sensitivity.csv'),('ablation',RESULTS / 'out_ablation.csv')]:
    J[k]=pd.read_csv(f).round(4).to_dict('records')
J['summary']=json.load(open(RESULTS / 'out_summary.json')); J['latency']=json.load(open(RESULTS / 'out_latency_summary.json'))
J['mech_n']=len(pd.read_csv(RESULTS / 'out_mechanism.csv'))
mlf=CACHE / 'ml.pkl'
ml=pickle.load(open(mlf,'rb')); mrows=[]
for (m,s),r in ml.items():
    for b,(p,df) in r.items(): mrows.append(dict(model=m,seed=s,b=b,cov=p['cov'],cov_n=p['cov_n'],far=p['far'],act=p['act']))
MR=pd.DataFrame(mrows); J['ml']=MR.groupby(['model','b']).agg(cov=('cov','mean'),cov_sd=('cov','std'),cov_n0=('cov_n','first'),far=('far','mean'),act=('act','mean'),n=('seed','count')).reset_index().round(4).to_dict('records')
json.dump(J,open(RESULTS / 'results.json','w'),ensure_ascii=False,indent=1,default=float); print('ok')
