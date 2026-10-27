import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_engine, list_plants
from core.forecast import fetch, RUN_BASE
from pipeline import save_day

TAG = sys.argv[2] if len(sys.argv) > 2 else "1600"
days = int(sys.argv[1]) if len(sys.argv) > 1 else 4
today = pd.Timestamp(datetime.now(timezone(timedelta(hours=9))).date()) + pd.Timedelta(days=1)
dates = [today - pd.Timedelta(days=k) for k in range(days - 1, -1, -1)]
plants = list_plants()
grids = sorted({(int(r.nx), int(r.ny)) for r in plants.itertuples()})

q = """select target_date from forecasts_hourly where run_tag = :t
       group by target_date having count(distinct plant_id) >= :n"""
with get_engine().connect() as c:
    done = {r[0] for r in c.execute(text(q), {"t": TAG, "n": len(plants)})}
todo = [d for d in dates if d.date() not in done]
print(f"대상 {len(dates)}일 중 남은 {len(todo)}일", flush=True)


def warm(t):
    try:
        fetch(*t)
    except Exception:
        pass


fmt = lambda d: d.strftime("%Y%m%d")
fails = 0
for i in range(0, len(todo), 30):
    chunk = todo[i:i + 30]
    tasks = set()
    for d in chunk:
        for g in grids:
            tasks.add((*g, fmt(d - pd.Timedelta(days=3)), "2300"))
            tasks.add((*g, fmt(d - pd.Timedelta(days=2)), "2300"))
            tasks.add((*g, fmt(d - pd.Timedelta(days=1)), RUN_BASE[TAG]))
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(warm, tasks))
    for d in chunk:
        ok, tot = save_day(d, TAG, plants)
        print(d.date(), f"{ok}/{tot}", flush=True)
        fails = fails + 1 if ok == 0 else 0
        if fails >= 3:
            print("3일 연속 실패 → 중단 (API 한도나 접속 확인)")
            sys.exit(1)
print("완료")