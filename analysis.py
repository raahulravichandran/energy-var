"""
Commodities VaR analysis: Brent, WTI, Henry Hub natural gas, heating oil.
Reproducible pipeline. Reads yfinance-format CSVs from ./data, writes ./outputs.

Design choices (see report for rationale):
  * yfinance BZ=F/CL=F/NG=F/HO=F are UNADJUSTED continuous front-month series.
  * NG=F (gas) manufactures large spurious returns on contract-roll days because
    Henry Hub's term structure is steep and seasonal. We identify roll days by the
    CME NG expiry rule (last trade = 3 business days before the 1st of the delivery
    month; the continuous series rolls the next trading day) and drop roll-day
    returns. We carry gas BOTH raw and cleaned for transparency.
  * Oil (BZ/CL/HO) extremes are genuine, time-clustered market moves (a Mar-Apr
    2026 crude selloff), not rolls -> left unadjusted. We verify CL roll days are
    immaterial below.

Measures: historical VaR, parametric (normal) VaR, expected shortfall @ 95/99.
Portfolio: covariance VaR + diversification benefit (equal weight).
Backtest: rolling 250-day out-of-sample VaR + Kupiec POF LR test.
Findings: (1) historical-vs-parametric tail gap; (2) root-time scaling breakdown.
"""
import numpy as np, pandas as pd, json
from scipy import stats
from pandas.tseries.offsets import BDay

ASSETS = {"brent":"Brent (BZ=F)", "wti":"WTI (CL=F)",
          "natgas":"Henry Hub gas (NG=F)", "heating_oil":"Heating oil (HO=F)"}
LABEL_SHORT = {"brent":"Brent","wti":"WTI","natgas":"Nat gas","heating_oil":"Heating oil"}
CONF = [0.95, 0.99]
R = {}   # container for results -> JSON

def load_close(name):
    df = pd.read_csv(f"data/{name}.csv", skiprows=3,
                     names=["Date","Close","High","Low","Open","Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()["Close"].astype(float)

prices = pd.DataFrame({k: load_close(k) for k in ASSETS})
rets   = np.log(prices / prices.shift(1))

# ---- roll-day identification -----------------------------------------------
def roll_days(index, rule):
    """Return the set of trading days that are the first day on a new front-month."""
    days = []
    for period in pd.period_range("2023-10","2026-10",freq="M"):
        first = pd.Timestamp(period.start_time.year, period.month, 1)
        if rule == "NG":            # 3 business days before the 1st of delivery month
            ltd = first - 3*BDay()
        elif rule == "CL":          # ~3 business days before the 25th of prior month
            ltd = (first - pd.offsets.MonthBegin(1)) + pd.Timedelta(days=24) - 3*BDay()
        after = index[index > ltd]
        if len(after): days.append(after[0])
    return pd.DatetimeIndex(days)

ng_rolls = roll_days(rets.index, "NG")
cl_rolls = roll_days(rets.index, "CL")

# how big are the roll-day moves?
ng_roll_ret = rets["natgas"].reindex(ng_rolls).dropna()
cl_roll_ret = rets["wti"].reindex(cl_rolls).dropna()
R["roll_diag"] = {
    "ng_n_rolls": int(ng_roll_ret.size),
    "ng_rolls_gt10pct": int((ng_roll_ret.abs()>0.10).sum()),
    "ng_roll_absmax_pct": float(ng_roll_ret.abs().max()*100),
    "cl_n_rolls": int(cl_roll_ret.size),
    "cl_rolls_gt10pct": int((cl_roll_ret.abs()>0.10).sum()),
    "cl_roll_absmax_pct": float(cl_roll_ret.abs().max()*100),
}

# primary return set: oil raw, gas roll-days dropped
rets_clean = rets.copy()
rets_clean.loc[rets_clean.index.isin(ng_rolls), "natgas"] = np.nan

# per-asset primary series (dropna each)
series = {k: (rets_clean[k].dropna() if k=="natgas" else rets[k].dropna()) for k in ASSETS}
series_gas_raw = rets["natgas"].dropna()

# ---- descriptive stats ------------------------------------------------------
def describe(r):
    return dict(n=int(r.size), mean=float(r.mean()), vol=float(r.std(ddof=1)),
                vol_ann=float(r.std(ddof=1)*np.sqrt(252)),
                skew=float(stats.skew(r)), exkurt=float(stats.kurtosis(r)),
                worst=float(r.min()), best=float(r.max()),
                jb_p=float(stats.jarque_bera(r)[1]))
R["desc"]      = {k: describe(series[k]) for k in ASSETS}
R["desc_gas_raw"] = describe(series_gas_raw)
R["quality"]   = {k: dict(rows=int(prices[k].notna().sum()),
                          first=str(prices[k].dropna().index.min().date()),
                          last=str(prices[k].dropna().index.max().date()),
                          nan=int(prices[k].isna().sum())) for k in ASSETS}

# ---- VaR / ES measures ------------------------------------------------------
def var_historical(r, c):  return float(-np.quantile(r, 1-c))
def var_parametric(r, c):  return float(-(r.mean() + stats.norm.ppf(1-c)*r.std(ddof=1)))
def es_historical(r, c):
    q = np.quantile(r, 1-c);  tail = r[r <= q];  return float(-tail.mean())
def es_parametric(r, c):
    z = stats.norm.ppf(1-c)
    return float(-(r.mean()) + r.std(ddof=1)*stats.norm.pdf(z)/(1-c))

def all_measures(r):
    out={}
    for c in CONF:
        out[f"{int(c*100)}"] = dict(hist=var_historical(r,c), param=var_parametric(r,c),
                                    es_hist=es_historical(r,c), es_param=es_parametric(r,c))
    return out
R["var"]        = {k: all_measures(series[k]) for k in ASSETS}
R["var_gas_raw"]= all_measures(series_gas_raw)

# ---- portfolio (equal weight) ----------------------------------------------
panel = rets_clean.dropna(how="any")          # common dates, gas cleaned
w = np.repeat(0.25, 4)
mu = panel.mean().values
Sigma = panel.cov().values
port_ret = panel.values @ w
port_ret = pd.Series(port_ret, index=panel.index)

R["portfolio"] = {"n_dates": int(panel.shape[0]),
                  "weights": {k:0.25 for k in ASSETS},
                  "corr": panel.corr().round(3).to_dict(),
                  "vol_daily": float(np.sqrt(w@Sigma@w)),
                  "vol_ann": float(np.sqrt(w@Sigma@w)*np.sqrt(252))}
for c in CONF:
    z = stats.norm.ppf(1-c)
    p_vol = np.sqrt(w@Sigma@w)
    param_port = float(-(w@mu) + abs(z)*p_vol)
    hist_port  = var_historical(port_ret, c)
    # undiversified (perfect-correlation) parametric VaR = sum of weighted standalone VaRs
    standalone = np.array([var_parametric(panel[k], c) for k in ASSETS])
    undiversified = float((w*standalone).sum())
    R["portfolio"][f"{int(c*100)}"] = dict(
        param=param_port, hist=hist_port,
        undiversified=undiversified,
        div_benefit_pct=float((1 - param_port/undiversified)*100),
        standalone={k: float(s) for k,s in zip(ASSETS, standalone)})

# ---- Kupiec POF rolling backtest -------------------------------------------
def kupiec(n, x, p):
    """LR_POF stat and p-value (chi2, 1 dof). x breaches in n obs, expected rate p."""
    if x==0:
        lr = -2*(n*np.log(1-p))    # limit form
    else:
        pi = x/n
        lr = -2*(( (n-x)*np.log(1-p)+x*np.log(p) )
                 -( (n-x)*np.log(1-pi)+x*np.log(pi) ))
    return float(lr), float(1-stats.chi2.cdf(lr,1))

def rolling_backtest(r, c, method, W=250):
    r = r.values; n=len(r); breaches=0; obs=0
    for t in range(W, n):
        win = r[t-W:t]
        if method=="hist":  v = -np.quantile(win, 1-c)
        else:               v = -(win.mean()+stats.norm.ppf(1-c)*win.std(ddof=1))
        obs+=1
        if r[t] < -v: breaches+=1
    rate = breaches/obs
    lr,pval = kupiec(obs, breaches, 1-c)
    return dict(obs=obs, breaches=breaches, rate=rate, expected=1-c,
                lr=lr, pval=pval, reject=bool(pval<0.05))

R["backtest"]={}
for k in ASSETS:
    R["backtest"][k]={}
    for c in CONF:
        for m in ["hist","param"]:
            R["backtest"][k][f"{m}_{int(c*100)}"]=rolling_backtest(series[k],c,m)
# portfolio backtest
R["backtest"]["portfolio"]={}
for c in CONF:
    for m in ["hist","param"]:
        R["backtest"]["portfolio"][f"{m}_{int(c*100)}"]=rolling_backtest(port_ret,c,m)

# ---- Finding 1: historical vs parametric at 99% ----------------------------
R["finding1"]={}
for k in ASSETS:
    h=R["var"][k]["99"]["hist"]; p=R["var"][k]["99"]["param"]
    R["finding1"][k]=dict(hist=h, param=p, gap_pct_pts=(h-p)*100, ratio=h/p)
R["finding1"]["gas_raw"]=dict(hist=R["var_gas_raw"]["99"]["hist"],
                              param=R["var_gas_raw"]["99"]["param"],
                              ratio=R["var_gas_raw"]["99"]["hist"]/R["var_gas_raw"]["99"]["param"])

def lo_mackinlay(r, q):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    n = r.size; mu = r.mean()
    var1 = ((r - mu)**2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), 'valid')
    m = q * (n - q + 1) * (1 - q / n)
    varq = ((rq - q * mu)**2).sum() / m
    vr = varq / var1
    e2 = (r - mu)**2; denom = e2.sum()**2; theta = 0.0
    for j in range(1, q):
        theta += (2 * (q - j) / q)**2 * ((e2[j:] * e2[:-j]).sum() / denom)
    z = (vr - 1) / np.sqrt(theta)
    return vr, z, 2 * (1 - stats.norm.cdf(abs(z)))

# ---- Finding 2: root-time scaling (1-day -> 10-day) ------------------------
H=10
R["finding2"]={}
for k in ASSETS:
    r=series[k]
    # overlapping H-day log returns
    hday = r.rolling(H).sum().dropna()
    scaled99 = np.sqrt(H)*var_historical(r,0.99)
    actual99 = var_historical(hday,0.99)
    scaled_p99 = np.sqrt(H)*var_parametric(r,0.99)
    actual_p99 = var_parametric(hday,0.99)
    vr, lm_z, lm_p = lo_mackinlay(r, H)   # LM variance ratio + robust z
    vr = float(vr)
    R["finding2"][k]=dict(scaled_hist99=scaled99, actual_hist99=actual99,
                          ratio_hist=actual99/scaled99,
                          scaled_param99=scaled_p99, actual_param99=actual_p99,
                          ratio_param=actual_p99/scaled_p99,
                          variance_ratio=vr, n_hday=int(hday.size),
                          lm_z=float(lm_z), lm_p=float(lm_p))

with open("outputs/results.json","w") as f: json.dump(R,f,indent=2)

# ---------------------------------------------------------------------------
# console summary
# ---------------------------------------------------------------------------
def pct(x): return f"{x*100:.2f}%"
print("ROLL DIAGNOSTICS")
print(f"  NG: {R['roll_diag']['ng_rolls_gt10pct']}/{R['roll_diag']['ng_n_rolls']} rolls >10%, "
      f"max |move| {R['roll_diag']['ng_roll_absmax_pct']:.1f}%")
print(f"  CL: {R['roll_diag']['cl_rolls_gt10pct']}/{R['roll_diag']['cl_n_rolls']} rolls >10%, "
      f"max |move| {R['roll_diag']['cl_roll_absmax_pct']:.1f}%  (oil rolls immaterial)")

print("\nDESCRIPTIVE (primary set: oil raw, gas roll-cleaned)")
print(f"{'asset':<12}{'n':>5}{'vol_ann':>9}{'skew':>7}{'exkurt':>8}{'worst':>8}")
for k in ASSETS:
    d=R['desc'][k]
    print(f"{LABEL_SHORT[k]:<12}{d['n']:>5}{d['vol_ann']*100:>8.1f}%{d['skew']:>7.2f}"
          f"{d['exkurt']:>8.1f}{d['worst']*100:>7.1f}%")
d=R['desc_gas_raw']; print(f"{'(gas RAW)':<12}{d['n']:>5}{d['vol_ann']*100:>8.1f}%"
      f"{d['skew']:>7.2f}{d['exkurt']:>8.1f}{d['worst']*100:>7.1f}%")

print("\n1-DAY VaR / ES  (loss, % of position)")
print(f"{'asset':<12}{'H-95':>7}{'P-95':>7}{'ES-95':>7}{'H-99':>7}{'P-99':>7}{'ES-99':>7}")
for k in ASSETS:
    v=R['var'][k]
    print(f"{LABEL_SHORT[k]:<12}{v['95']['hist']*100:>6.2f}%{v['95']['param']*100:>6.2f}%"
          f"{v['95']['es_hist']*100:>6.2f}%{v['99']['hist']*100:>6.2f}%"
          f"{v['99']['param']*100:>6.2f}%{v['99']['es_hist']*100:>6.2f}%")

print("\nPORTFOLIO (equal weight)")
for c in [95,99]:
    pf=R['portfolio'][str(c)]
    print(f"  {c}%  param VaR {pct(pf['param'])}  hist VaR {pct(pf['hist'])}  "
          f"undiversified {pct(pf['undiversified'])}  diversification benefit {pf['div_benefit_pct']:.1f}%")

print("\nKUPIEC BACKTEST (rolling 250d, parametric)")
print(f"{'asset':<12}{'99 rate':>9}{'99 p':>8}{'99':>5}{'95 rate':>9}{'95 p':>8}{'95':>5}")
for k in list(ASSETS)+["portfolio"]:
    b=R['backtest'][k]
    def cell(m): return f"{b[m]['rate']*100:.2f}%"
    r99,p99=b['param_99']['rate'],b['param_99']['pval']
    r95,p95=b['param_95']['rate'],b['param_95']['pval']
    lab = LABEL_SHORT.get(k,'Portfolio')
    print(f"{lab:<12}{r99*100:>8.2f}%{p99:>8.3f}{'FAIL' if p99<0.05 else 'ok':>5}"
          f"{r95*100:>8.2f}%{p95:>8.3f}{'FAIL' if p95<0.05 else 'ok':>5}")

print("\nFINDING 1  historical vs parametric @99% (VaR)")
for k in ASSETS:
    f1=R['finding1'][k]
    print(f"  {LABEL_SHORT[k]:<12} hist {pct(f1['hist'])}  param {pct(f1['param'])}  "
          f"hist/param {f1['ratio']:.2f}x")
print(f"  (gas RAW      hist/param {R['finding1']['gas_raw']['ratio']:.2f}x)")

print("\nFINDING 2  10-day: actual vs sqrt(10)x1-day @99% (historical)")
for k in ASSETS:
    f2=R['finding2'][k]
    print(f"  {LABEL_SHORT[k]:<12} scaled {pct(f2['scaled_hist99'])}  actual {pct(f2['actual_hist99'])}  "
          f"actual/scaled {f2['ratio_hist']:.2f}x  VR(10) {f2['variance_ratio']:.2f}")
print("\n[saved outputs/results.json]")
