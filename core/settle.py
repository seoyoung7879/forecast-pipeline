from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
RESID_CSV = ROOT / "데이터" / "잔차풀_행단위.csv"
TIER_4, TIER_3, MIN_UTIL, MIN_N = 4.0, 6.0, 0.10, 30


def load_resid(path=RESID_CSV):
    d = pd.read_csv(path, encoding="utf-8-sig")[["plant", "year", "month", "day", "hour", "잔차"]].dropna()
    pool = {}
    for m in range(1, 13):
        win = [(m - 2) % 12 + 1, m, m % 12 + 1]
        for h, g in d[d["month"].isin(win)].groupby("hour"):
            pool[(m, int(h))] = np.sort(g["잔차"].values)
    z = np.zeros(len(d))
    for (m, h), idx in d.groupby(["month", "hour"]).indices.items():
        a = pool[(m, int(h))]
        u = (np.searchsorted(a, d["잔차"].values[idx], side="right") - 0.5) / len(a)
        z[idx] = norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
    d["z"] = z
    pv = d.pivot_table(index=["plant", "year", "month", "day"], columns="hour", values="z")
    c = pv.corr(min_periods=MIN_N).values
    rho = float(np.clip(np.nanmean(c[~np.eye(len(c), dtype=bool)]), 0.0, 0.95))
    return dict(pool=pool, rho=rho)


def get_pool(res, m, h):
    return res["pool"].get((m, int(h)))
def price_of(err_pct, util):
    return np.where(err_pct <= TIER_4, 4.0, np.where(err_pct <= TIER_3, 3.0, 0.0)) * (util >= MIN_UTIL)


def simulate_pool(P, cap, dm, month, res, scenario="corr", B=400, seed=7):
    n, rho = len(cap), res["rho"]
    rng = np.random.default_rng(seed)
    Y = np.zeros((B, n, 24))
    zd, zh = rng.standard_normal((B, 1)), rng.standard_normal((B, 24))
    for i in range(n):
        if scenario != "corr" and i > 0:
            zd, zh = rng.standard_normal((B, 1)), rng.standard_normal((B, 24))
        U = norm.cdf(np.sqrt(rho) * zd + np.sqrt(1 - rho) * zh)
        E = np.zeros((B, 24))
        for h in np.where(dm[i])[0]:
            arr = get_pool(res, month, h)
            if arr is None:
                continue
            E[:, h] = arr[np.minimum((U[:, h] * len(arr)).astype(int), len(arr) - 1)]
        Y[:, i, :] = np.clip(P[i] + E, 0, 1) * dm[i]
    sumY = (Y * cap[None, :, None]).sum(1)
    sumP = (P * dm * cap[:, None]).sum(0)
    cs = cap.sum()
    err = np.abs(sumY - sumP[None, :]) / cs * 100
    return dict(gen=sumY * 1000, money=sumY * 1000 * price_of(err, sumY / cs))