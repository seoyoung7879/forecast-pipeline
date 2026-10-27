import pandas as pd
from sqlalchemy import text
from db import get_engine

q = """select target_date, run_tag, count(*) n, count(distinct plant_id) plants
       from forecasts_hourly group by 1, 2 order by 1, 2"""
with get_engine().connect() as c:
    print(pd.read_sql(text(q), c))