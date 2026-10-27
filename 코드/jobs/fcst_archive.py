import sys, os, time
from pathlib import Path
from datetime import datetime, timedelta
import requests
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import get_engine
import core.forecast  # .env 로드
KEY = os.getenv("SOLAR_API_KEY")
URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
BASE_TIMES = ["0200","0500","0800","1100","1400","1700","2000","2300"]
eng = get_engine()

with eng.begin() as c:
    c.execute(text("""create table if not exists forecast_raw(
        nx int, ny int, base_date text, base_time text,
        fcst_date text, fcst_time text, category text, value text,
        primary key (nx, ny, base_date, base_time, fcst_date, fcst_time, category))"""))
    grids = c.execute(text("select distinct nx, ny from plants")).fetchall()

def fetch(nx, ny, bd, bt):
    p = dict(serviceKey=KEY, pageNo="1", numOfRows="1500", dataType="JSON",
             base_date=bd, base_time=bt, nx=nx, ny=ny)
    j = requests.get(URL, params=p, timeout=30).json()["response"]
    if j["header"]["resultCode"] != "00":
        return None
    return j["body"]["items"]["item"]

SQL = text("""insert into forecast_raw values
  (:nx,:ny,:bd,:bt,:fd,:ft,:cat,:val) on conflict do nothing""")

now = datetime.now()
for d in range(0, 4):
    bd = (now - timedelta(days=d)).strftime("%Y%m%d")
    for bt in BASE_TIMES:
        if bd == now.strftime("%Y%m%d") and now.strftime("%H%M") < str(int(bt) + 10).zfill(4):
            continue
        n = 0
        for nx, ny in grids:
            items = fetch(nx, ny, bd, bt)
            if not items:
                continue
            rows = [dict(nx=nx, ny=ny, bd=bd, bt=bt, fd=i["fcstDate"], ft=i["fcstTime"],
                         cat=i["category"], val=str(i["fcstValue"])) for i in items]
            with eng.begin() as c:
                c.execute(SQL, rows)
            n += 1
            time.sleep(0.2)
        print(bd, bt, "격자", n, "/", len(grids))