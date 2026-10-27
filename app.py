import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent
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
    with st.form("add"):
        a1, a2, a3 = st.columns(3)
        name = a1.text_input("이름")
        cap_in = a2.number_input("설비용량(MW)", 0.1, 50.0, 1.0, 0.1)
        built = a3.date_input("준공일", value=datetime(2022, 1, 1))
        b1, b2, b3 = st.columns(3)
        lat = b1.number_input("위도", 33.0, 39.0, 36.5, format="%.4f")
        lon = b2.number_input("경도", 124.0, 132.0, 127.5, format="%.4f")
        alt = b3.number_input("해발고도(m)", 0.0, 1500.0, 0.0)
        e1, e2, e3 = st.columns(3)
        sido, sigungu, emd = e1.text_input("시도"), e2.text_input("시군구"), e3.text_input("읍면동(선택)")
        go_add = st.form_submit_button("추가하고 내일 예측 생성")
    if go_add:
        if not name or not sido or not sigungu:
            st.error("이름, 시도, 시군구는 필수입니다.")
        else:
            nx, ny = to_grid(lat, lon)
            with st.spinner(f"격자({nx},{ny}) 예보로 예측 중..."):
                pid, ok, D, tg = create_plant(name=name, cap=cap_in, sido=sido, sigungu=sigungu,
                                              emd=emd or None, built=built, lat=lat, lon=lon,
                                              alt=alt, nx=nx, ny=ny)
            if ok:
                st.success(f"추가 완료 (id {pid}). {D} {tg} 예측이 저장되었습니다.")
            else:
                st.warning("발전소는 추가됐지만 예측 생성에 실패했습니다. 다음 자동 실행에서 예측됩니다.")
            st.rerun()