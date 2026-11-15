"""Step 7b. Cross-validation of the SCA component variants (Table 4)."""
import pickle, json
from config import CACHE, RESULTS
import numpy as np, pandas as pd
from cv import folds, SUBS
from core import wilson
A=pickle.load(open(CACHE / 'ablation_metrics.pkl', 'rb')); C=A['configs']; M=A['metrics']
piv=lambda col: M.pivot(index='cfg',columns='participant',values=col)[SUBS].values
T=dict(hit=piv('hit').astype(float),fa=piv('fa'),neg=piv('neg_time'),act=piv('act_time'))
def pooled(ci,si):
    h=T['hit'][np.ix_(ci,si)]; return np.nansum(h,1)/np.sum(~np.isnan(h),1), T['fa'][np.ix_(ci,si)].sum(1)/(T['neg'][np.ix_(ci,si)].sum(1)/3600), T['act'][np.ix_(ci,si)].sum(1)/T['neg'][np.ix_(ci,si)].sum(1)
def select(ci,tr,b,a=0.10):
    cov,far,act=pooled(ci,tr); feas=(far<=b)&(act<=a)
    if not feas.any(): return ci[np.argmin(far+100*np.maximum(act-a,0))]
    for i in np.lexsort((far,act,-cov)):
        if feas[i]: return ci[i]
variants={
 'Full SCA':lambda c:c[0]=='full',
 'w/o persistence (p = 0)':lambda c:c[0]=='full' and c[3]==0,
 'w/o lower bound (d0 = 0)':lambda c:c[0]=='nod0',
 'w/o ECD-boundary exclusion':lambda c:c[0]=='noexcl',
 'w/o duration mark (unit weight)':lambda c:c[0]=='nomark',
 'w/o time decay (event-indexed decay)':lambda c:c[0]=='evdecay',
 'Boxcar kernel instead of exponential':lambda c:c[0]=='boxcar',
 'Fixed λ = 60 s':lambda c:c[0]=='full' and c[2]==60,
}
n=len(SUBS); out=[]
for name,f in variants.items():
    ci=np.array([i for i,c in enumerate(C) if f(c)])
    for b in [2.5,5.0,8.0]:
        covs=[];fars=[];acts=[];s0=None
        for seed in range(50):
            hit=np.full(n,np.nan);fa=np.zeros(n);neg=np.zeros(n);act=np.zeros(n)
            for te in folds(seed):
                tr=np.setdiff1d(np.arange(n),te); c=select(ci,tr,b)
                hit[te]=T['hit'][c,te];fa[te]=T['fa'][c,te];neg[te]=T['neg'][c,te];act[te]=T['act'][c,te]
            cv=np.nansum(hit)/np.sum(~np.isnan(hit)); covs.append(cv);fars.append(fa.sum()/(neg.sum()/3600));acts.append(act.sum()/neg.sum())
            if seed==0: s0=int(np.nansum(hit))
        out.append(dict(variant=name,b=b,cov=np.mean(covs),cov_sd=np.std(covs),far=np.mean(fars),act=np.mean(acts),cov_n0=s0))
R=pd.DataFrame(out); R.to_csv(RESULTS / 'out_ablation.csv',index=False)
print(R.pivot(index='variant',columns='b',values='cov').round(3).loc[list(variants)])
print(R.pivot(index='variant',columns='b',values='far').round(2).loc[list(variants)])
print(R.pivot(index='variant',columns='b',values='act').round(3).loc[list(variants)])
