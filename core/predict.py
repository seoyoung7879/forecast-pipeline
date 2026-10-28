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

    # [추가] 날짜별 낮 마스크
    DAYm = np.zeros((3, 24), bool)
    for i, d in enumerate(days):
        g = f[f["date"] == d].set_index("hour")["태양고도"].reindex(range(24))
        DAYm[i] = g.values > 0

    ti_i = be["allc"].index(TI)
    # [삭제] ti_mu, ti_sd, te_mu, te_sd 두 줄
    mu_t, sd_t = be["mu"][TI], be["sd"][TI]

    irr = np.zeros((3, 24))
    for i in range(3):
        xi = build_x("irr", [i], prev, Dm, M, S, bi)
        y = ensemble(bi["nets"], xi, DAYm[[i]], "irr")[0] * M[i]   # [변경]
        irr[i] = y
        Dm[i, :, ti_i] = ((y - mu_t) / sd_t) * M[i]

    xe = build_x("eff", [2], prev, Dm, M, S, be)
    eff = ensemble(be["nets"], xe, DAYm[[2]], "eff")[0]            # [변경]

    dm = DAYm[2] & (M[2] > 0) & (irr[2] > 0)                       # [변경]
    return dict(p=eff * dm, dm=dm, irr=irr[2], base=last, missing=int(24 - M[2].sum()))