# Energy VaR

A 1-day Value-at-Risk study of a four-instrument energy book: Brent (BZ=F), WTI (CL=F), Henry Hub natural gas (NG=F) and heating oil (HO=F), 19 Sep 2023 to 18 Sep 2026 (755 daily observations).

Write-up: https://raahulravichandran.github.io/energy-var/

## Findings

The parametric 99% VaR fails its backtest on the oil complex. Over a rolling 250-day out-of-sample window, breach rates run 2.4% to 3.2% against a 1% target for Brent, WTI, heating oil and the equal-weight portfolio. Kupiec POF rejects all four at p < 0.05. Gas is the only book that passes.

Data hygiene decided the gas result before any model did. The free continuous NG=F series is unadjusted, so it books a fake return every time the front-month contract rolls. Fourteen of 36 roll days move more than 10%, the largest a spurious 64% log splice. Dropping roll-day returns by the CME last-trade rule cuts gas excess kurtosis from 35 to 7. WTI rolls were immaterial on the crude calendar, so oil is carried unadjusted.

Diversification in an energy book is almost entirely a gas story. The equal-weight 99% portfolio VaR is 5.3% against 6.9% undiversified, a 23.8% benefit. Brent, WTI and heating oil correlate 0.80 to 0.94, so for risk purposes they are one position in three tickers. Gas correlates 0.14 with crude.

## Method

Log returns. Historical (empirical quantile), parametric (normal) and expected shortfall at 95% and 99%. Portfolio VaR from the covariance matrix at equal weights. Rolling 250-day out-of-sample backtest with the Kupiec proportion-of-failures likelihood-ratio test. Horizon scaling tested with the heteroskedasticity-robust Lo-MacKinlay variance ratio at 10 days.

Every figure in the write-up is read from outputs/results.json, which analysis.py writes. No number is entered by hand.

## Reproduce

```
pip install -r requirements.txt
python analysis.py
python make_charts.py
```

analysis.py reads the four CSVs in data/ and writes outputs/results.json. make_charts.py reads that file and writes the six figures.

Tested on Python 3.9 with numpy 2.0.2, pandas 2.3.3, scipy 1.13.1 and matplotlib 3.9.4.

## Limitations

Roll calendars were tested for natural gas and WTI only. Brent and heating oil are carried unadjusted.

Three years spans one volatility regime shift, not many. It is enough to reject the normal assumption at 99% and to expose procyclicality in the rolling VaR. It is not enough to pin down 10-day scaling or the 99.9% tail.

Equal weights are a stand-in. The diversification figure is real but weight-dependent.

Single-name measures are computed on each series' own observations; portfolio measures on the common-date panel, so standalone figures differ marginally between the two.

## Data

Yahoo Finance daily settles, auto-adjusted close. Continuous front-month contracts.
