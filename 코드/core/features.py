import numpy as np, pandas as pd, pvlib

SKY, TI = "하늘상태", "일사(MJ/m2)"


def dew_point(t, rh):
    b, c = 17.62, 243.12
    rh = np.maximum(rh, 0.1)
    g = b * t / (c + t) + np.log(rh / 100.0)
    return c * g / (b - g)


def add_features(fc, lat, lon, alt, built, age_hi):
    f = fc.copy()
    dt = f["datetime"]
    f["hour"] = dt.dt.hour
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * f["hour"] / 24), np.cos(2 * np.pi * f["hour"] / 24)
    md = dt.dt.month + (dt.dt.day - 1) / dt.dt.days_in_month
    f["month_day_sin"], f["month_day_cos"] = np.sin(2 * np.pi * md / 12), np.cos(2 * np.pi * md / 12)
    f["이슬점"] = dew_point(f["기온(°C)"].values, f["습도(%)"].values)
    f["T-Td"] = f["기온(°C)"] - f["이슬점"]
    loc = pvlib.location.Location(latitude=lat, longitude=lon, tz="Asia/Seoul", altitude=alt)
    sp = loc.get_solarposition(times=pd.DatetimeIndex(dt).tz_localize("Asia/Seoul"))
    f["태양고도"], f["방위각"] = sp["apparent_elevation"].values, sp["azimuth"].values
    age = (dt.dt.normalize() - pd.Timestamp(built)).dt.days / 365.25
    f["연식(년)"] = age.clip(lower=0, upper=age_hi)
    f["date"] = dt.dt.normalize()
    return f


def make_arrays(f, days, b):
    allc, mu, sd = b["allc"], b["mu"], b["sd"]
    sky_map = {v: i for i, v in enumerate(b["meta"]["sky_vals"])}
    N = len(days)
    D = np.zeros((N, 24, len(allc)), np.float32)
    M = np.zeros((N, 24), np.float32)
    S = np.zeros((N, 24), np.int64)
    f = f.copy()
    f[TI] = 0.0
    Z = np.stack([(f[c].values.astype(float) - mu[c]) / sd[c] for c in allc], axis=1).astype(np.float32)
    di = f["date"].map({d: i for i, d in enumerate(days)})
    ok = di.notna().values
    di = di[ok].astype(int).values
    h = f["hour"].values[ok]
    D[di, h] = Z[ok]
    M[di, h] = 1
    S[di, h] = f.loc[ok, SKY].map(sky_map).fillna(0).astype(int).values
    prev = np.arange(N) - 1
    return D, M, S, prev


def build_x(task, idx, prev, D, M, S, b):
    cur, pv = b["meta"]["spec"][task]
    ci = [b["allc"].index(c) for c in cur]
    pi_ = [b["allc"].index(c) for c in pv]
    idx = np.asarray(idx)
    m = M[idx][:, :, None]
    sky = np.eye(len(b["meta"]["sky_vals"]), dtype=np.float32)[S[idx]] * m
    p = prev[idx]
    has = (p >= 0).astype(np.float32)
    pj = np.maximum(p, 0)
    x_prev = D[pj][:, :, pi_] * has[:, None, None]
    pm = (M[pj] * has[:, None])[:, :, None]
    X = np.concatenate([D[idx][:, :, ci], sky, m, x_prev, pm], axis=2)
    return np.ascontiguousarray(X.transpose(0, 2, 1))