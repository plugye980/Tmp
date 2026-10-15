import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
LAST_YEAR = 2025
MIN_DAYS = 300

# 디자인 토큰(web-test 다크 테마)
BG, SURFACE, WELL = "#212429", "#25282d", "#1c1f23"
TEXT, MUTED = "#e6e8eb", "#99a4b1"
SKY, BRASS = "#8cc7e8", "#e0b878"

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")
st.markdown(
    f"""
<style>
.stApp {{ background: linear-gradient(180deg, #1b1d21, {BG} 40%, #262a30); color: {TEXT};
  font-family: "Pretendard Variable", Pretendard, -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, p, label, span {{ color: {TEXT}; }}
.card {{ background: {SURFACE}; border-radius: 22px; padding: 1.4rem 1.6rem; margin: 0.8rem 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.055), 0 10px 24px rgba(0,0,0,0.55); }}
.well {{ background: {WELL}; border-radius: 22px; padding: 1rem 1.4rem; margin: 0.8rem 0;
  box-shadow: inset 0 2px 6px rgba(0,0,0,0.34); color: {MUTED}; font-size: 0.95rem; }}
.cap {{ color: {MUTED}; font-size: 0.85rem; letter-spacing: 0.04em; }}
.big {{ font-size: clamp(2.6rem, 8vw, 4.6rem); font-weight: 700; color: {BRASS}; line-height: 1.1; }}
.r {{ font-size: 1.8rem; font-weight: 600; color: {SKY}; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_yearly():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    yearly = df.groupby("연도")["평균기온"].agg(["mean", "count"]).reset_index()
    yearly = yearly[(yearly["연도"] <= LAST_YEAR) & (yearly["count"] >= MIN_DAYS)]
    return yearly.rename(columns={"mean": "연평균기온", "count": "관측일"})


yearly = load_yearly()
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy()
slope, intercept = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]

st.title("🌡️ 기온 예측기")
st.markdown(
    f'<div class="well">서울 연평균기온 회귀 직선 · 사용한 해 <b>{len(yearly)}개</b> · '
    f'시작 <b>{int(x.min())}년</b> · 끝 <b>{int(x.max())}년</b><br>'
    f'(2025년까지, 관측일 {MIN_DAYS}일 미만인 해 제외)</div>',
    unsafe_allow_html=True,
)

year = st.slider("연도", 1900, 2100, 2050)
pred = slope * year + intercept
st.markdown(
    f'<div class="card"><div class="cap">{year}년 예상 연평균기온</div>'
    f'<div class="big">{pred:.2f} ℃</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="card"><div class="cap">상관계수 r (연도–연평균기온)</div><div class="r">{r:.3f}</div>'
    f'<div class="cap">회귀식: 기온 = {slope:.4f} × 연도 {intercept:+.2f}</div></div>',
    unsafe_allow_html=True,
)

line_x = np.array([1900, 2100])
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="연평균기온",
                         marker=dict(color=SKY, size=7, opacity=0.8)))
fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept, mode="lines",
                         name="회귀 직선", line=dict(color=MUTED, width=2)))
fig.add_trace(go.Scatter(x=[year], y=[pred], mode="markers", name=f"{year}년 예측",
                         marker=dict(color=BRASS, size=14, symbol="star")))
fig.update_layout(
    paper_bgcolor=WELL, plot_bgcolor=WELL, font=dict(color=TEXT),
    xaxis=dict(title="연도", gridcolor="#2a3037", zeroline=False),
    yaxis=dict(title="연평균기온(℃)", gridcolor="#2a3037", zeroline=False),
    legend=dict(orientation="h", y=1.08), margin=dict(l=20, r=20, t=40, b=20), height=460,
)
st.plotly_chart(fig, width="stretch")
