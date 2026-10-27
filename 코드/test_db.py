from sqlalchemy import text
from sqlalchemy.engine import make_url
from db import get_engine

eng = get_engine()
u = make_url(str(eng.url))
print("host:", u.host, "| port:", u.port, "| user:", u.username, "| db:", u.database)
try:
    with eng.connect() as c:
        print(c.execute(text("select version()")).scalar())
        q = "select table_name from information_schema.tables where table_schema='public' order by 1"
        print([r[0] for r in c.execute(text(q))])
except Exception as e:
    print("ERROR:", str(e).splitlines()[0])
    print("ERROR2:", str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else "")