import numpy as np, pandas as pd
from core.model import load_models, ensemble
from core.forecast import collect
from core.features import add_features, make_arrays, build_x, TI


def predict_plant_day(p, D, tag):
    bi, be = load_models()
    D = pd.Timestamp(D).normalize()
    fc, last = collect(int(p["nx"]), int(p["ny"]), D, tag)
    if fc is None:
        return None
    age_hi = float(be["mu"]["연식(년)"] + 3 * be["sd"]["연식(년)"])
    f = add_features(fc, float(p["lat"]), float(p["lon"]), float(p["alt"]), p["built_date"], age_hi)
    days = [D - pd.Timedelta(days=2), D - pd.Timedelta(days=1), D]
    Dm, M, S, prev = make_arrays(f, days, be)

    ti_i = be["allc"].index(TI)
    ti_mu, ti_sd = bi["meta"]["ti_mu"], bi["meta"]["ti_sd"]
    te_mu, te_sd = be["meta"]["te_mu"], be["meta"]["te_sd"]
    mu_t, sd_t = be["mu"][TI], be["sd"][TI]

    irr = np.zeros((3, 24))
    for i in range(3):
        xi = build_x("irr", [i], prev, Dm, M, S, bi)
        y = np.clip(ensemble(bi["nets"], xi)[0] * ti_sd + ti_mu, 0, None) * M[i]
        irr[i] = y
        Dm[i, :, ti_i] = ((y - mu_t) / sd_t) * M[i]

    xe = build_x("eff", [2], prev, Dm, M, S, be)
    eff = np.clip(ensemble(be["nets"], xe)[0] * te_sd + te_mu, 0, None)

    fd = f[f["date"] == D].set_index("hour")["태양고도"].reindex(range(24))
    dm = (fd.values > 0) & (M[2] > 0) & (irr[2] > 0)
    return dict(p=eff * dm, dm=dm, irr=irr[2], base=last, missing=int(24 - M[2].sum()))