import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import list_plants
import pandas as pd

pd.set_option("display.width", 200)
df = list_plants()
print(df[["id", "name", "capacity_mw", "sigungu", "eupmyeondong", "built_date", "nx", "ny", "active"]].to_string(index=False))
print(len(df), "개 | 합계", round(df["capacity_mw"].astype(float).sum(), 2), "MW | 격자", df[["nx", "ny"]].drop_duplicates().shape[0], "개")