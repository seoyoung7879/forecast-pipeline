import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import list_plants, reset_plants, add_plants_bulk

ROOT = Path(__file__).resolve().parents[2]
g = pd.read_csv(ROOT / "데이터" / "격자예보_관측지점매핑.csv").dropna(subset=["위도", "경도", "격자 X", "격자 Y"])

REGIONS = [("강원", "고성군"), ("제주", "제주시"), ("경상남|경남", "고성군"), ("전라남|전남", "영암군"),
           ("충청남|충남", "당진시"), ("경상북|경북", "영천시"), ("전라북|전북", "익산시"), ("경기", "이천시")]

n_old = len(list_plants(active_only=False))
if n_old > 0:
    if input(f"기존 {n_old}개를 지우고 다시 만들까요? (y/n) ").strip().lower() != "y":
        sys.exit()

picked = []
for a, b in REGIONS:
    r = g[g["1단계"].str.contains(a) & (g["2단계"] == b)]
    r2 = r.dropna(subset=["3단계"])
    r = r2 if not r2.empty else r
    if r.empty:
        print("매핑에서 못 찾음:", a, b)
    else:
        picked.append(r.iloc[0])
assert picked, "지역을 하나도 못 찾았습니다"

rng = np.random.RandomState(7)
caps = np.round(rng.uniform(0.5, 2.0, 24), 1)
rows = []
for k in range(24):
    r = picked[k % len(picked)]
    built = (pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.randint(0, 2190)))).date()
    emd = None if pd.isna(r["3단계"]) else str(r["3단계"])
    rows.append(dict(name=f"가상태양광{k + 1:02d}호", cap=float(caps[k]), sido=str(r["1단계"]), sigungu=str(r["2단계"]),
                     emd=emd, built=built, lat=float(r["위도"]), lon=float(r["경도"]),
                     alt=float(np.nan_to_num(r.get("노장해발고도(m)", 0.0))), nx=int(r["격자 X"]), ny=int(r["격자 Y"])))

assert sum(x["cap"] for x in rows) > 20, "합계가 20MW 이하입니다"
reset_plants()
add_plants_bulk(rows)
df = list_plants()
print(len(df), "개 | 합계", round(df["capacity_mw"].astype(float).sum(), 1), "MW |",
      df[["nx", "ny"]].drop_duplicates().shape[0], "개 격자")