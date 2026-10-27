import pandas as pd
from datetime import date, timedelta
from db import list_plants
from core.model import load_models
from core.forecast import collect
from core.features import add_features, make_arrays, build_x

bi, be = load_models()
p = list_plants().iloc[1]
D = pd.Timestamp(date.today() + timedelta(days=1))
fc, _ = collect(int(p["nx"]), int(p["ny"]), D, "1000")
age_hi = float(be["mu"]["연식(년)"] + 3 * be["sd"]["연식(년)"])
f = add_features(fc, float(p["lat"]), float(p["lon"]), float(p["alt"]), p["built_date"], age_hi)
days = [D - pd.Timedelta(days=2), D - pd.Timedelta(days=1), D]
Dm, M, S, prev = make_arrays(f, days, be)
Xi = build_x("irr", [0, 1, 2], prev, Dm, M, S, bi)
Xe = build_x("eff", [0, 1, 2], prev, Dm, M, S, be)
print(Dm.shape, Xi.shape, Xe.shape, "| 연식 상한", round(age_hi, 2))
print(f[["기온(°C)", "태양고도", "연식(년)"]].describe().loc[["min", "max"]])