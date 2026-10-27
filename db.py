import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def get_engine():
    host = os.getenv("DB_HOST")
    if not host:
        raise RuntimeError("DB_HOST가 .env에 없습니다")
    url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=host,
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "postgres"),
    )
    return create_engine(url, pool_pre_ping=True)
from sqlalchemy import text
import pandas as pd


def list_plants(active_only=True):
    q = "select * from plants" + (" where active" if active_only else "") + " order by id"
    with get_engine().connect() as c:
        return pd.read_sql(text(q), c)


def add_plant(name, cap, sido, sigungu, emd, built, lat, lon, alt, nx, ny):
    q = text("""insert into plants
        (name, capacity_mw, sido, sigungu, eupmyeondong, built_date, lat, lon, alt, nx, ny)
        values (:name, :cap, :sido, :sigungu, :emd, :built, :lat, :lon, :alt, :nx, :ny)
        returning id""")
    p = dict(name=name, cap=float(cap), sido=sido, sigungu=sigungu, emd=emd, built=built,
             lat=float(lat), lon=float(lon), alt=float(alt), nx=int(nx), ny=int(ny))
    with get_engine().begin() as c:
        return c.execute(q, p).scalar()


def deactivate_plant(pid):
    with get_engine().begin() as c:
        c.execute(text("update plants set active=false where id=:i"), {"i": int(pid)})
def reset_plants():
    with get_engine().begin() as c:
        c.execute(text("truncate plants restart identity cascade"))


def add_plants_bulk(rows):
    q = text("""insert into plants
        (name, capacity_mw, sido, sigungu, eupmyeondong, built_date, lat, lon, alt, nx, ny)
        values (:name, :cap, :sido, :sigungu, :emd, :built, :lat, :lon, :alt, :nx, :ny)""")
    with get_engine().begin() as c:
        for p in rows:
            c.execute(q, p)
def upsert_forecasts(rows):
    q = text("""insert into forecasts_hourly
        (plant_id, target_date, run_tag, hour, p, dm, base_date, base_time)
        values (:plant_id, :target_date, :run_tag, :hour, :p, :dm, :base_date, :base_time)
        on conflict (plant_id, target_date, run_tag, hour) do update set
        p = excluded.p, dm = excluded.dm, base_date = excluded.base_date,
        base_time = excluded.base_time, created_at = now()""")
    with get_engine().begin() as c:
        c.execute(q, rows)