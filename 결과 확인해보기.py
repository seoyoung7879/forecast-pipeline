import sqlite3
import pandas as pd

conn = sqlite3.connect("predictions.db")
df = pd.read_sql("SELECT * FROM predictions", conn)
conn.close()

print(df['발전구분'].value_counts())
print(df[['발전구분', 'datetime', '예측효율', '예측발전량(kWh)']].head(10))