import os, re, time
from functools import lru_cache
from pathlib import Path
import pandas as pd, requests
from dotenv import load_dotenv
from streamlit import code

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
API_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
RUN_BASE = {"1000": "0800", "1600": "1400"}
SKY = "하늘상태"
COLS = ["기온(°C)", "강수량(mm)", "습도(%)", "풍속(m/s)", SKY]


def _pcp(v):
    s = str(v)
    if "없음" in s: return 0.0
    if "미만" in s: return 0.5
    x = [float(t) for t in re.findall(r"\d+\.?\d*", s)]
    return sum(x) / len(x) if x else 0.0


@lru_cache(maxsize=None)
def fetch(nx, ny, base_date, base_time):
    params = dict(serviceKey=os.getenv("SOLAR_API_KEY"), pageNo="1", numOfRows="1000", dataType="JSON",
                  base_date=base_date, base_time=base_time, nx=str(nx), ny=str(ny))
    last = None
    for k in range(3):
        try:
            r = requests.get(API_URL, params=params, timeout=30)
            r.raise_for_status()
            body = r.json()["response"]
            code = body["header"]["resultCode"]
            if code == "03":
                return None
            code = body["header"]["resultCode"]
            msg = body["header"].get("resultMsg", "")
            if code == "03" or "최근 3일" in msg:
                return None
            if code != "00":
                raise RuntimeError(msg)
            if code != "00":
                raise RuntimeError(body["header"]["resultMsg"])
            items = body["body"]["items"]["item"]
            break
        except Exception as e:
            last = e
            time.sleep(2 * (k + 1))
    else:
        raise RuntimeError(f"예보 수집 실패 {nx},{ny},{base_date},{base_time}: {last}")
    raw = pd.DataFrame(items)
    raw["val"] = [_pcp(v) if c == "PCP" else pd.to_numeric(v, errors="coerce") for c, v in zip(raw["category"], raw["fcstValue"])]
    fc = raw.pivot_table(index=["fcstDate", "fcstTime"], columns="category", values="val").reset_index()
    fc = fc.rename(columns={"TMP": "기온(°C)", "PCP": "강수량(mm)", "REH": "습도(%)", "WSD": "풍속(m/s)", "SKY": SKY})
    for c in COLS:
        if c not in fc:
            fc[c] = 0.0
    fc["강수량(mm)"] = fc["강수량(mm)"].fillna(0)
    fc["datetime"] = pd.to_datetime(fc["fcstDate"] + fc["fcstTime"], format="%Y%m%d%H%M")
    return fc[["datetime"] + COLS]


def collect(nx, ny, D, tag):
    D = pd.Timestamp(D).normalize()
    f = lambda d: d.strftime("%Y%m%d")
    jobs = [(f(D - pd.Timedelta(days=3)), "2300"), (f(D - pd.Timedelta(days=2)), "2300"),
            (f(D - pd.Timedelta(days=1)), RUN_BASE[tag])]
    frames = [x.copy() for x in (fetch(nx, ny, bd, bt) for bd, bt in jobs) if x is not None]
    if not frames:
        return None, jobs[-1]
    fc = pd.concat(frames).drop_duplicates("datetime", keep="last").sort_values("datetime")
    fc = fc[fc["datetime"].dt.normalize().between(D - pd.Timedelta(days=2), D)]
    return fc.reset_index(drop=True), jobs[-1]