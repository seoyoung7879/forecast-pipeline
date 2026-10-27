import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import list_plants, upsert_forecasts
from core.predict import predict_plant_day


def save_day(D, tag, plants=None):
    D = pd.Timestamp(D).normalize()
    plants = list_plants() if plants is None else plants
    rows, ok = [], 0
    for _, p in plants.iterrows():
        try:
            r = predict_plant_day(p, D, tag)
        except Exception as e:
            print("실패:", p["name"], str(e)[:100])
            continue
        if r is None:
            continue
        ok += 1
        for h in range(24):
            rows.append(dict(plant_id=int(p["id"]), target_date=D.date(), run_tag=tag, hour=h,
                             p=float(r["p"][h]), dm=bool(r["dm"][h]),
                             base_date=r["base"][0], base_time=r["base"][1]))
    if rows:
        upsert_forecasts(rows)
    return ok, len(plants)