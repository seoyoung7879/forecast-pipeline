from datetime import date, timedelta
import pandas as pd
from core.forecast import collect

D = pd.Timestamp(date.today() + timedelta(days=1))
fc, last_job = collect(53, 38, D, "1000")
print("마지막 발표:", last_job)
print(None if fc is None else fc.shape)
if fc is not None:
    print(fc.groupby(fc["datetime"].dt.date).size())
    print(fc.tail(3))