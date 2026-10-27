from datetime import date, timedelta
import numpy as np, pandas as pd
from db import list_plants
from core.predict import predict_plant_day
from core.settle import load_resid, simulate_pool

res = load_resid()
sizes = pd.Series({k: len(v) for k, v in res["pool"].items()})
print("ρ =", round(res["rho"], 3), "| 칸", len(sizes), "| 표본 최소/중앙/최대", sizes.min(), int(sizes.median()), sizes.max())
miss = {m: [h for h in range(7, 19) if (m, h) not in res["pool"]] for m in range(1, 13)}
print("7~18시 중 빈 칸:", {m: v for m, v in miss.items() if v})

D = pd.Timestamp(date.today() + timedelta(days=1))
pl = list_plants().iloc[[1, 4, 7]]
rs = [predict_plant_day(p, D, "1000") for _, p in pl.iterrows()]
P, dm = np.array([r["p"] for r in rs]), np.array([r["dm"] for r in rs])
cap = pl["capacity_mw"].astype(float).values
for sc in ("corr", "indep"):
    s = simulate_pool(P, cap, dm, D.month, res, sc)
    print(sc, "| 발전량(kWh)", round(s["gen"].sum(1).mean()), "| 정산금(원)", round(s["money"].sum(1).mean()))
print(sizes[sizes < 30].sort_index())