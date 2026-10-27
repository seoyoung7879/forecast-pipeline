import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent
GRID_CSV = ROOT / "데이터" / "격자예보_관측지점매핑.csv"

@st.cache_data
def grid_df():
    return pd.read_csv(GRID_CSV, encoding="utf-8-sig").dropna(subset=["위도", "경도", "격자 X", "격자 Y"])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))
from db import get_engine, list_plants, add_plant, deactivate_plant
from core.settle import load_resid, simulate_pool

KST = timezone(timedelta(hours=9))
TAGS = {"1600": "전일 16시 시점 예보", "1000": "전일 10시 시점 예보"}
st.set_page_config(page_title="VPP 발전량·정산금 예측", page_icon="☀️", layout="wide")


def to_grid(lat, lon):
    D = np.pi / 180
    re = 6371.00877 / 5.0
    s1, s2, ol, oa = 30 * D, 60 * D, 126 * D, 38 * D
    sn = np.log(np.cos(s1) / np.cos(s2)) / np.log(np.tan(np.pi / 4 + s2 / 2) / np.tan(np.pi / 4 + s1 / 2))
    sf = np.tan(np.pi / 4 + s1 / 2) ** sn * np.cos(s1) / sn
    ro = re * sf / np.tan(np.pi / 4 + oa / 2) ** sn
    ra = re * sf / np.tan(np.pi / 4 + lat * D / 2) ** sn
    th = ((lon * D - ol + np.pi) % (2 * np.pi) - np.pi) * sn
    return int(ra * np.sin(th) + 43 + 0.5), int(ro - ra * np.cos(th) + 136 + 0.5)


@st.cache_resource
def get_res():
    return load_resid()

@st.cache_data(ttl=300)
def plants_all():
    return list_plants(active_only=False)

@st.cache_data(ttl=300)
def avail_dates(tag):
    q = "select distinct target_date from forecasts_hourly where run_tag=:t order by 1 desc"
    with get_engine().connect() as c:
        return [r[0] for r in c.execute(text(q), {"t": tag})]

@st.cache_data(ttl=300)
def load_fc(d, tag):
    q = "select plant_id, hour, p, dm from forecasts_hourly where target_date=:d and run_tag=:t"
    with get_engine().connect() as c:
        return pd.read_sql(text(q), c, params={"d": d, "t": tag})

@st.cache_data(show_spinner="정산 시뮬레이션 중...")
def run_sim(P, cap, DM, month):
    return simulate_pool(P, cap, DM, month, get_res(), "corr")


def remove_plant(pid):
    today = datetime.now(KST).date()
    deactivate_plant(pid)
    with get_engine().begin() as c:
        c.execute(text("delete from forecasts_hourly where plant_id=:i and target_date > :d"),
                  {"i": int(pid), "d": today})
    st.cache_data.clear()


def create_plant(**kw):
    from pipeline import save_day
    pid = add_plant(**kw)
    st.cache_data.clear()
    now = datetime.now(KST)
    tag = "1000" if now.hour < 12 else "1600"
    D = (now + timedelta(days=1)).date()
    row = list_plants(active_only=False).query("id == @pid")
    ok, _ = save_day(D, tag, row)
    st.cache_data.clear()
    return pid, ok, D, tag


plants = plants_all()
with st.sidebar:
    st.header("조회 조건")
    tag = st.selectbox("예보 시점", list(TAGS), format_func=TAGS.get)
    ds = avail_dates(tag)
    day = st.selectbox("대상일", ds) if ds else None

if day is None:
    st.warning("저장된 예측이 없습니다.")
    st.stop()

fc = load_fc(day, tag)
sel = plants[plants["id"].isin(set(fc["plant_id"]))].sort_values("id").reset_index(drop=True)
@st.cache_data(ttl=3600, show_spinner="연간 예상 수익 계산 중...")
def annual_estimate(tag, n_dates, last_date):
    q = "select target_date, plant_id, hour, p, dm from forecasts_hourly where run_tag=:t"
    with get_engine().connect() as c:
        df = pd.read_sql(text(q), c, params={"t": tag})
    capmap = plants_all().set_index("id")["capacity_mw"].astype(float)
    rows = []
    for d, g in df.groupby("target_date"):
        ids = sorted(g["plant_id"].unique())
        piv = lambda col: g.pivot(index="plant_id", columns="hour", values=col).reindex(
            index=ids, columns=range(24))
        P = piv("p").fillna(0).values.astype(float)
        DM = piv("dm").fillna(False).astype(bool).values
        o = simulate_pool(P, capmap.loc[ids].values, DM, pd.Timestamp(d).month, get_res(), "corr", B=200)
        rows.append((d, float(o["money"].sum(1).mean()), float(o["gen"].sum(1).mean())))
    return pd.DataFrame(rows, columns=["date", "money", "kwh"]).sort_values("date")
st.title("☀️ VPP 발전량·예상 정산금")
tab1, tab2, tab3 = st.tabs(["VPP 전체", "개별 발전소", "VPP 관리"])

if not sel.empty:
    pv = lambda col: fc.pivot(index="plant_id", columns="hour", values=col).reindex(
        index=sel["id"], columns=range(24))
    P = pv("p").fillna(0).values.astype(float)
    DM = pv("dm").fillna(False).values.astype(bool)
    cap = sel["capacity_mw"].astype(float).values
    month = pd.Timestamp(day).month
    out = run_sim(P, cap, DM, month)
    gen, money = out["gen"], out["money"]
    gen_day, money_day = gen.sum(1), money.sum(1)
    exp_gen = (P * DM * cap[:, None]).sum(0) * 1000
    hrs = list(range(24))

    with tab1:
        st.caption(f"{day} · {TAGS[tag]} · 당시 구성 발전소 {len(sel)}개 · 합계 {cap.sum():.1f} MW")
        if cap.sum() <= 20:
            st.warning("설비 합계가 20MW 이하라 참고용입니다.")
        c1, c2, c3 = st.columns(3)
        est = annual_estimate(tag, len(ds), str(ds[0]))
        est = est.tail(365)
        n = len(est)
        done = est["money"].sum()
        avg = est["money"].mean()
        annual = done + avg * max(365 - n, 0)

        st.subheader("예상 연간 수익")
        y1, y2, y3 = st.columns(3)
        y1.metric("예상 연간 수익", f"{annual:,.0f} 원")
        y2.metric("하루 평균 정산금", f"{avg:,.0f} 원")
        y3.metric("예측 반영 일수", f"{n} / 365일")
        st.caption(f"예측이 있는 {n}일의 정산금 합계({done:,.0f}원)에 "
           f"나머지 {max(365 - n, 0)}일을 하루 평균으로 채운 값입니다. "
           "그날그날의 VPP 구성 기준이라, 발전소를 바꾸면 이후 날짜부터 반영됩니다.")
        if n < 30:
            st.warning("예측 일수가 적어 계절 변동이 반영되지 않았습니다. 데이터가 쌓일수록 정확해집니다.")
        c1.metric("총 예상 발전량", f"{exp_gen.sum():,.0f} kWh",
                  f"90% 구간 {np.percentile(gen_day,5):,.0f}~{np.percentile(gen_day,95):,.0f}")
        c2.metric("예상 정산금(기대값)", f"{money_day.mean():,.0f} 원",
                  f"90% 구간 {np.percentile(money_day,5):,.0f}~{np.percentile(money_day,95):,.0f}")
        c3.metric("평균 효율(낮)", f"{(P*DM).sum()/max(DM.sum(),1)*100:.1f} %")

        lo, hi = np.percentile(gen, [5, 95], axis=0)
        f1 = go.Figure()
        f1.add_scatter(x=hrs, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip")
        f1.add_scatter(x=hrs, y=lo, fill="tonexty", fillcolor="rgba(255,167,38,0.25)",
                       line=dict(width=0), name="90% 구간")
        f1.add_scatter(x=hrs, y=exp_gen, mode="lines+markers", name="예상 발전량",
                       line=dict(color="#F57C00", width=3))
        f1.update_layout(height=380, xaxis_title="시", yaxis_title="kWh", hovermode="x unified")
        st.subheader("시간대별 발전량")
        st.plotly_chart(f1, use_container_width=True)

        st.subheader("시간대별 예상 정산금")
        f2 = go.Figure(go.Bar(x=hrs, y=money.mean(0), marker_color="#42A5F5"))
        f2.update_layout(height=320, xaxis_title="시", yaxis_title="원(기대값)")
        st.plotly_chart(f2, use_container_width=True)

        tbl = pd.DataFrame({"발전소": sel["name"], "설비(MW)": cap,
                            "예상 발전량(kWh)": (P * DM).sum(1) * cap * 1000,
                            "지역": sel["sigungu"], "준공일": sel["built_date"]})
        st.subheader("발전소별 요약")
        st.dataframe(tbl.round(1), hide_index=True, use_container_width=True)

    with tab2:
        pick = st.selectbox("발전소", list(sel["name"]))
        i = int(sel.index[sel["name"] == pick][0])
        kwh = P[i] * DM[i] * cap[i] * 1000
        solo = run_sim(P[i:i+1], cap[i:i+1], DM[i:i+1], month)
        m1, m2 = st.columns(2)
        m1.metric("예상 발전량", f"{kwh.sum():,.0f} kWh")
        m2.metric("단독 기대 정산금", f"{solo['money'].sum(1).mean():,.0f} 원")
        st.caption(f"{sel.loc[i,'sido']} {sel.loc[i,'sigungu']} · 설비 {cap[i]} MW · 준공 {sel.loc[i,'built_date']}")
        f3 = go.Figure(go.Bar(x=hrs, y=kwh, marker_color="#F57C00"))
        f3.update_layout(height=340, xaxis_title="시", yaxis_title="kWh")
        st.plotly_chart(f3, use_container_width=True)
else:
    with tab1:
        st.warning("이 날짜·시점에는 예측이 없습니다.")

with tab3:
    st.subheader("현재 VPP (앞으로의 예측 기준)")
    active = plants[plants["active"]].reset_index(drop=True)
    st.caption(f"활성 발전소 {len(active)}개 · 합계 {active['capacity_mw'].astype(float).sum():.1f} MW. "
               "변경은 내일 예측부터 적용되고, 오늘 이전 예측은 그대로 유지됩니다.")
    st.dataframe(active[["id", "name", "capacity_mw", "sigungu", "eupmyeondong", "built_date", "nx", "ny"]],
                 hide_index=True, use_container_width=True)

    st.subheader("발전소 삭제")
    d1, d2 = st.columns([3, 1])
    target = d1.selectbox("삭제할 발전소", list(active["id"]),
                          format_func=lambda i: active.set_index("id").loc[i, "name"])
    sure = d2.checkbox("삭제 확인")
    if st.button("삭제", disabled=not sure):
        remove_plant(target)
        st.success("삭제했습니다. 내일 이후 예측에서 제외됩니다.")
        st.rerun()

    st.subheader("발전소 추가")
    g = grid_df()
    a1, a2, a3 = st.columns(3)
    name = a1.text_input("이름")
    cap_in = a2.number_input("설비용량(MW)", 0.1, 50.0, 1.0, 0.1)
    built = a3.date_input("준공일", value=datetime(2022, 1, 1))

    e1, e2, e3 = st.columns(3)
    sido = e1.selectbox("시도", sorted(g["1단계"].unique()))
    g1 = g[g["1단계"] == sido]
    sigungu = e2.selectbox("시군구", sorted(g1["2단계"].dropna().unique()))
    g2 = g1[g1["2단계"] == sigungu]
    emds = sorted(g2["3단계"].dropna().unique())
    emd = e3.selectbox("읍면동", emds) if emds else None
    row = (g2[g2["3단계"] == emd] if emd else g2).iloc[0]
    nx, ny = int(row["격자 X"]), int(row["격자 Y"])
    lat, lon = float(row["위도"]), float(row["경도"])
    alt = row.get("노장해발고도(m)")

    if st.button("추가하고 내일 예측 생성"):
        if not name.strip():
            st.error("이름을 입력하세요.")
        elif name.strip() in set(plants["name"]):
            st.error("이미 있는 이름입니다.")
        elif pd.isna(alt):
            st.error("이 지역은 해발고도 정보가 없어 선택할 수 없습니다. 다른 지역을 고르세요.")
        else:
            with st.spinner(f"격자({nx},{ny}) 예보로 예측 중..."):
                pid, ok, D, tg = create_plant(name=name.strip(), cap=cap_in, sido=sido, sigungu=sigungu,
                                              emd=emd, built=built, lat=lat, lon=lon,
                                              alt=float(alt), nx=nx, ny=ny)
            if ok:
                st.success(f"추가 완료 (id {pid}). {D} {tg} 예측이 저장되었습니다.")
            else:
                st.warning("발전소는 추가됐지만 예측 생성에 실패했습니다. 다음 자동 실행에서 예측됩니다.")
            st.rerun()