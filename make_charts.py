"""Generate report figures (matplotlib) -> outputs/*.png, following the dataviz palette."""
import numpy as np, pandas as pd, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from pandas.tseries.offsets import BDay

# ---- palette (validated dataviz reference) ----
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; VIOLET="#4a3aa7"
RED="#e34948"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8a86"; SURF="#fcfcfb"; GRID="#e6e5e1"
ASSET_C={"brent":BLUE,"wti":ORANGE,"natgas":AQUA,"heating_oil":VIOLET}
SH={"brent":"Brent","wti":"WTI","natgas":"Nat gas","heating_oil":"Heating oil"}

plt.rcParams.update({
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "font.size":11,"axes.edgecolor":INK2,"axes.linewidth":0.8,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.color":INK2,"ytick.color":INK2,"text.color":INK,"axes.labelcolor":INK2,
    "font.family":"DejaVu Sans",
})

# ---- rebuild data ----
def load_close(name):
    df=pd.read_csv(f"data/{name}.csv",skiprows=3,names=["Date","Close","High","Low","Open","Volume"])
    df["Date"]=pd.to_datetime(df["Date"]); return df.set_index("Date").sort_index()["Close"].astype(float)
prices=pd.DataFrame({k:load_close(k) for k in ASSET_C})
rets=np.log(prices/prices.shift(1))
def rolls(idx):
    ds=[]
    for period in pd.period_range("2023-10","2026-10",freq="M"):
        first=pd.Timestamp(period.start_time.year,period.month,1); a=idx[idx>first-3*BDay()]
        if len(a): ds.append(a[0])
    return pd.DatetimeIndex(ds)
ng_rolls=rolls(rets.index)
rc=rets.copy(); rc.loc[rc.index.isin(ng_rolls),"natgas"]=np.nan
series={k:(rc[k].dropna() if k=="natgas" else rets[k].dropna()) for k in ASSET_C}
gas_raw=rets["natgas"].dropna()
R=json.load(open("outputs/results.json"))

def var_h(r,c): return -np.quantile(r,1-c)
def var_p(r,c): return -(r.mean()+stats.norm.ppf(1-c)*r.std(ddof=1))

# ============================================================== FIG 1: roll artifact
fig,ax=plt.subplots(1,2,figsize=(11,4.0),gridspec_kw={"width_ratios":[1.25,1]})
rr=gas_raw.reindex(ng_rolls).dropna()*100
colors=[RED if abs(v)>10 else MUTED for v in rr]
ax[0].bar(range(len(rr)),rr.values,color=colors,width=0.8)
ax[0].axhline(10,ls="--",lw=1,color=INK2); ax[0].axhline(-10,ls="--",lw=1,color=INK2)
ax[0].set_title("Gas 'returns' on the 36 contract-roll days",fontweight="bold",color=INK,loc="left")
ax[0].set_ylabel("roll-day log return (%)"); ax[0].set_xlabel("monthly roll (Oct 2023 → Sep 2026)")
ax[0].text(0.5,-58,"14 of 36 rolls exceed ±10%  —  spurious splices, not market moves",
           color=RED,fontsize=9.5)
ax[0].set_xticks([])
# right: distribution raw vs cleaned (zoom on left tail)
bins=np.linspace(-0.30,0.30,80)
ax[1].hist(gas_raw,bins=bins,color=MUTED,alpha=0.55,label=f"raw  (kurt {R['desc_gas_raw']['exkurt']:.0f})")
ax[1].hist(series["natgas"],bins=bins,color=AQUA,alpha=0.75,label=f"roll-cleaned  (kurt {R['desc']['natgas']['exkurt']:.0f})")
ax[1].set_title("Gas daily-return distribution",fontweight="bold",color=INK,loc="left")
ax[1].set_xlabel("daily log return"); ax[1].set_ylabel("days"); ax[1].legend(frameon=False,fontsize=9)
ax[1].set_xlim(-0.30,0.30)
plt.tight_layout(); plt.savefig("outputs/fig1_rolls.png",dpi=150); plt.close()

# ============================================================== FIG 2: distributions vs normal
fig,axes=plt.subplots(2,2,figsize=(11,7.2))
for ax,k in zip(axes.ravel(),ASSET_C):
    r=series[k]; mu,sd=r.mean(),r.std(ddof=1)
    lo,hi=np.quantile(r,0.001),np.quantile(r,0.999)
    rng=max(abs(lo),abs(hi))*1.1
    bins=np.linspace(-rng,rng,70)
    ax.hist(r,bins=bins,density=True,color=ASSET_C[k],alpha=0.35)
    xs=np.linspace(-rng,rng,400)
    ax.plot(xs,stats.norm.pdf(xs,mu,sd),color=INK,lw=1.6,label="normal fit")
    vh=-var_h(r,0.99); vp=-var_p(r,0.99)
    ax.axvline(vh,color=ASSET_C[k],lw=2,label=f"hist 99% VaR {abs(vh)*100:.1f}%")
    ax.axvline(vp,color=RED,lw=2,ls="--",label=f"param 99% VaR {abs(vp)*100:.1f}%")
    ax.set_title(f"{SH[k]}   (excess kurtosis {R['desc'][k]['exkurt']:.1f})",
                 fontweight="bold",color=INK,loc="left",fontsize=11)
    ax.set_yticks([]); ax.set_xlim(-rng,rng); ax.legend(frameon=False,fontsize=8,loc="upper left")
    ax.set_xlabel("daily log return")
fig.suptitle("Return distributions vs the normal assumption  (left tail = loss)",
             fontweight="bold",fontsize=13,color=INK,x=0.01,ha="left")
plt.tight_layout(rect=[0,0,1,0.97]); plt.savefig("outputs/fig2_dist.png",dpi=150); plt.close()

# ============================================================== FIG 3: VaR measures @99
fig,ax=plt.subplots(figsize=(11,4.2))
ks=list(ASSET_C); x=np.arange(len(ks)); w=0.26
h=[R["var"][k]["99"]["hist"]*100 for k in ks]
p=[R["var"][k]["99"]["param"]*100 for k in ks]
e=[R["var"][k]["99"]["es_hist"]*100 for k in ks]
b1=ax.bar(x-w,h,w,color=BLUE,label="Historical VaR")
b2=ax.bar(x,p,w,color=ORANGE,label="Parametric VaR (normal)")
b3=ax.bar(x+w,e,w,color=VIOLET,label="Expected shortfall")
for bars in (b1,b2,b3):
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.15,f"{bar.get_height():.1f}",
                ha="center",va="bottom",fontsize=8.5,color=INK2)
ax.set_xticks(x); ax.set_xticklabels([SH[k] for k in ks])
ax.set_ylabel("1-day loss, % of position"); ax.legend(frameon=False,ncol=3,fontsize=9.5,loc="upper left")
ax.set_title("99% risk measures by instrument",fontweight="bold",color=INK,loc="left")
ax.set_ylim(0,15)
plt.tight_layout(); plt.savefig("outputs/fig3_var.png",dpi=150); plt.close()

# ============================================================== FIG 4: backtest breach rates @99
fig,ax=plt.subplots(figsize=(11,4.2))
ks2=ks+["portfolio"]; labs=[SH[k] for k in ks]+["Portfolio"]
x=np.arange(len(ks2)); w=0.36
pr=[R["backtest"][k]["param_99"]["rate"]*100 for k in ks2]
hr=[R["backtest"][k]["hist_99"]["rate"]*100 for k in ks2]
bp=ax.bar(x-w/2,pr,w,color=ORANGE,label="Parametric 99% VaR")
bh=ax.bar(x+w/2,hr,w,color=BLUE,label="Historical 99% VaR")
ax.axhline(1.0,color=INK,lw=1.5,ls="--"); ax.text(len(ks2)-0.4,1.08,"1% expected",fontsize=9,color=INK)
for bars,rates,key,meth in [(bp,pr,ks2,"param"),(bh,hr,ks2,"hist")]:
    for bar,k in zip(bars,ks2):
        pval=R["backtest"][k][f"{meth}_99"]["pval"]
        mark="✕" if pval<0.05 else "✓"
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,mark,ha="center",
                va="bottom",fontsize=10,color=RED if pval<0.05 else "#008300",fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set_ylabel("observed breach rate (%)"); ax.legend(frameon=False,fontsize=9.5,loc="upper right")
ax.set_title("Kupiec backtest: 99% VaR breach rates vs the 1% target   (✕ = model rejected, p<0.05)",
             fontweight="bold",color=INK,loc="left",fontsize=11.5)
ax.set_ylim(0,3.7)
plt.tight_layout(); plt.savefig("outputs/fig4_backtest.png",dpi=150); plt.close()

# ============================================================== FIG 5: correlation heatmap
fig,ax=plt.subplots(figsize=(5.6,4.8))
C=pd.DataFrame(R["portfolio"]["corr"]).loc[ks,ks]
im=ax.imshow(C.values,cmap="BuPu",vmin=0,vmax=1)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels([SH[k] for k in ks],rotation=30,ha="right"); ax.set_yticklabels([SH[k] for k in ks])
for i in range(4):
    for j in range(4):
        v=C.values[i,j]
        ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                color="white" if v>0.55 else INK,fontsize=11,fontweight="bold")
ax.set_title("Return correlations",fontweight="bold",color=INK,loc="left")
cb=plt.colorbar(im,fraction=0.046,pad=0.04); cb.outline.set_visible(False)
plt.tight_layout(); plt.savefig("outputs/fig5_corr.png",dpi=150); plt.close()

# ============================================================== FIG 6: variance ratio scaling
fig,ax=plt.subplots(figsize=(8.2,4.2))
vrs=[R["finding2"][k]["variance_ratio"] for k in ks]
zs=[R["finding2"][k]["lm_z"] for k in ks]
# 2SE band: since z=(vr-1)/se -> se=(vr-1)/z
ses=[abs((vr-1)/z) if abs(z)>1e-6 else np.nan for vr,z in zip(vrs,zs)]
y=np.arange(len(ks))
ax.axvline(1.0,color=INK,lw=1.5,ls="--"); ax.text(1.005,3.35,"√t holds (VR=1)",fontsize=9,color=INK)
for i,(vr,se,k) in enumerate(zip(vrs,ses,ks)):
    ax.errorbar(vr,i,xerr=2*se,fmt="o",color=ASSET_C[k],ecolor=ASSET_C[k],
                elinewidth=2,capsize=4,markersize=9)
    ax.text(vr,i+0.18,f"VR={vr:.2f}",ha="center",fontsize=9,color=INK2)
ax.set_yticks(y); ax.set_yticklabels([SH[k] for k in ks]); ax.set_ylim(-0.6,3.7)
ax.set_xlabel("10-day variance ratio  (Var₁₀ / 10·Var₁),  ±2 SE (Lo–MacKinlay)")
ax.set_title("Does √t scaling hold at 10 days?  Every band straddles 1 — cannot reject √t",
             fontweight="bold",color=INK,loc="left",fontsize=11)
ax.set_xlim(0.4,1.7)
plt.tight_layout(); plt.savefig("outputs/fig6_scaling.png",dpi=150); plt.close()

print("charts written:", [f for f in __import__("os").listdir("outputs") if f.endswith(".png")])
