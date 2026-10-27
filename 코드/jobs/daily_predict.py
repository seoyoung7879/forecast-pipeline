import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import save_day

now = datetime.now(timezone(timedelta(hours=9)))
tag = sys.argv[1] if len(sys.argv) > 1 else ("1000" if now.hour < 12 else "1600")
D = (now + timedelta(days=1)).date()
ok, tot = save_day(D, tag)
print(f"{D} {tag}: {ok}/{tot} 발전소 저장")
if ok < tot:
    sys.exit(1)