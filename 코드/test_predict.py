from datetime import date, timedelta
import numpy as np, pandas as pd
from db import list_plants
from core.predict import predict_plant_day

D = pd.Timestamp(date.today() + timedelta(days=1))
for _, p in list_plants().iloc[[1, 4]].iterrows():
    r = predict_plant_day(p, D, "1000")
    print(p["name"], p["sigungu"], r["base"], "| 결측", r["missing"], "| 낮", int(r["dm"].sum()), "시간")
    print("일사(MJ/m2):", np.round(r["irr"][6:19], 2))
    print("효율:", np.round(r["p"][6:19], 3))