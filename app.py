import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import google.generativeai as genai
from data import (
    COUNTRIES, REGIONS, GPI_TRENDS, DOMAIN_TRENDS,
    ECONOMIC_IMPACT, INDICATORS, TOP_IMPROVERS, TOP_DETERIORATORS, SUMMARY
)

st.set_page_config(
    page_title="Global Peace Index 2025",
    page_icon="🕊️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Gemini setup ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

gemini_model = get_gemini()

# ── Data prep ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.DataFrame(COUNTRIES)

    def peace_tier(s):
        if s < 1.5:  return "🟢 Very High Peace"
        if s < 2.0:  return "🟡 High Peace"
        if s < 2.5:  return "🟠 Medium Peace"
        if s < 3.0:  return "🔴 Low Peace"
        return "⛔ Very Low Peace"

    df["peaceTier"]       = df["score"].apply(peace_tier)
    df["changeDirection"] = df["yearChange"].apply(
        lambda x: "Improved" if x < -0.005 else ("Deteriorated" if x > 0.005 else "No Change")
    )
    df["changeMagnitude"] = df["yearChange"].abs()
    df["changePct"]       = (df["yearChange"] / df["score"] * 100).round(2)
    return df

df_full = load_data()
trend_df   = pd.DataFrame(GPI_TRENDS)
domain_df  = pd.DataFrame(DOMAIN_TRENDS)
econ_df    = pd.DataFrame(ECONOMIC_IMPACT)
region_df  = pd.DataFrame(REGIONS)
ind_df     = pd.DataFrame(INDICATORS)

SCORE_COLOR = px.colors.diverging.RdYlGn[::-1]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕊️ GPI 2025 Dashboard")
    st.caption("Institute for Economics & Peace")
    st.markdown("---")
    st.markdown("### 🔍 Filters")

    # 1. Country search
    search = st.text_input("1️⃣ Search Country", placeholder="e.g. Iceland, India…")

    # 2. Region
    all_regions = sorted(df_full["region"].unique().tolist())
    sel_regions = st.multiselect("2️⃣ Region", all_regions, default=all_regions)

    # 3. Peace tier
    all_tiers = ["🟢 Very High Peace","🟡 High Peace","🟠 Medium Peace","🔴 Low Peace","⛔ Very Low Peace"]
    sel_tiers = st.multiselect("3️⃣ Peace Tier", all_tiers, default=all_tiers)

    # 4. GPI score range
    score_min, score_max = st.slider("4️⃣ GPI Score Range", 1.0, 4.0, (1.0, 4.0), 0.01)

    # 5. Rank range
    rank_min, rank_max = st.slider("5️⃣ Rank Range", 1, 163, (1, 163))

    # 6. Change direction
    change_dir = st.selectbox("6️⃣ Change Direction", ["All","Improved","Deteriorated","No Change"])

    # 7. Change magnitude threshold
    change_mag = st.slider("7️⃣ Max Change Magnitude", 0.0, 0.20, 0.20, 0.005,
                           help="Filter to countries with small or large year-over-year changes")

    st.markdown("---")
    st.markdown("##### Domain Score Ranges")

    # 8. Safety score
    saf_min, saf_max = st.slider("8️⃣ Safety Score", 1.0, 4.0, (1.0, 4.0), 0.01)

    # 9. Conflict score
    con_min, con_max = st.slider("9️⃣ Conflict Score", 1.0, 4.0, (1.0, 4.0), 0.01)

    # 10. Militarisation score
    mil_min, mil_max = st.slider("🔟 Militarisation Score", 1.0, 4.0, (1.0, 4.0), 0.01)

    st.markdown("---")
    st.markdown("##### Display Options")

    # 11. Top N for rankings chart
    top_n = st.slider("1️⃣1️⃣ Top N in Rankings", 5, 50, 20)

    # 12. Sort by
    sort_by = st.selectbox("1️⃣2️⃣ Sort Table By",
        ["rank","score","safetyScore","conflictScore","militarisationScore","yearChange","country"])

    # 13. Sort direction
    sort_asc = st.radio("1️⃣3️⃣ Sort Direction", ["Ascending","Descending"], horizontal=True) == "Ascending"

    # 14. Domain focus for scatter
    domain_focus = st.selectbox("1️⃣4️⃣ Domain Focus (Scatter)", 
        ["Safety & Security","Ongoing Conflict","Militarisation"])

    # 15. Compare specific countries
    compare_countries = st.multiselect("1️⃣5️⃣ Highlight Countries",
        sorted(df_full["country"].tolist()),
        help="Selected countries will be highlighted across charts")

    # 16. High-conflict filter
    conflict_only = st.checkbox("1️⃣6️⃣ High-Conflict Only (conflict score > 2.5)")

    # 17. Deviation from global average
    global_avg = df_full["score"].mean()
    deviation = st.slider("1️⃣7️⃣ Max Deviation from Global Avg (±)",
                          0.0, 2.5, 2.5, 0.1,
                          help=f"Global average is {global_avg:.3f}")

    # 18. Color scheme
    color_scheme = st.radio("1️⃣8️⃣ Map Color Scheme",
        ["By GPI Score","By Change Direction","By Region"], horizontal=False)

    st.markdown("---")
    st.caption("Data: IEP Global Peace Index 2025")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full.copy()

if search:
    df = df[df["country"].str.contains(search, case=False, na=False)]
if sel_regions:
    df = df[df["region"].isin(sel_regions)]
if sel_tiers:
    df = df[df["peaceTier"].isin(sel_tiers)]

df = df[
    (df["score"]               >= score_min) & (df["score"]               <= score_max) &
    (df["rank"]                >= rank_min)  & (df["rank"]                <= rank_max)  &
    (df["safetyScore"]         >= saf_min)   & (df["safetyScore"]         <= saf_max)   &
    (df["conflictScore"]       >= con_min)   & (df["conflictScore"]       <= con_max)   &
    (df["militarisationScore"] >= mil_min)   & (df["militarisationScore"] <= mil_max)   &
    (df["changeMagnitude"]     <= change_mag)
]

if change_dir != "All":
    df = df[df["changeDirection"] == change_dir]
if conflict_only:
    df = df[df["conflictScore"] > 2.5]

df = df[abs(df["score"] - global_avg) <= deviation]
df = df.sort_values(sort_by, ascending=sort_asc)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🕊️ Global Peace Index 2025")
st.markdown("**Comprehensive peace analytics • 163 countries • Institute for Economics & Peace**")

if len(df) < len(df_full):
    st.info(f"🔎 Showing **{len(df)}** of **{len(df_full)}** countries based on active filters.")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("🌍 Countries Monitored", f"{SUMMARY['totalCountries']}")
k2.metric("⚔️ Active Conflicts", f"{SUMMARY['activeConflicts']}", help="Most since WWII")
k3.metric("💰 Economic Cost", f"${SUMMARY['economicImpactTrillions']}T", help="11.6% of global GDP")
k4.metric("📉 Global Score Change", f"{SUMMARY['globalScoreChange']:+.2f}%", delta=f"{SUMMARY['globalScoreChange']:+.2f}%", delta_color="inverse")
k5.metric("✅ Countries Improved", f"{SUMMARY['countriesImproved']}")
k6.metric("❌ Countries Deteriorated", f"{SUMMARY['countriesDeteriored']}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️ World Map", "📊 Rankings", "📈 Trends",
    "🌍 Regions", "🔬 Analysis", "📋 Data Table", "🤖 AI Assistant"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — World Map
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Global Peace Map")
    col1, col2 = st.columns([3, 1])

    with col1:
        if color_scheme == "By GPI Score":
            fig_map = px.choropleth(
                df, locations="country", locationmode="country names",
                color="score", color_continuous_scale="RdYlGn_r",
                range_color=[1.0, 4.0],
                hover_name="country",
                hover_data={"rank": True, "score": ":.3f",
                            "safetyScore": ":.3f", "conflictScore": ":.3f",
                            "militarisationScore": ":.3f", "yearChange": ":.3f"},
                title="GPI Score by Country (lower = more peaceful)",
                labels={"score": "GPI Score"},
                height=500,
            )
        elif color_scheme == "By Change Direction":
            color_map = {"Improved": "#22c55e", "Deteriorated": "#ef4444", "No Change": "#94a3b8"}
            fig_map = px.choropleth(
                df, locations="country", locationmode="country names",
                color="changeDirection",
                color_discrete_map=color_map,
                hover_name="country",
                hover_data={"rank": True, "score": ":.3f", "yearChange": ":.3f"},
                title="Year-over-Year Change Direction",
                height=500,
            )
        else:
            fig_map = px.choropleth(
                df, locations="country", locationmode="country names",
                color="region",
                hover_name="country",
                hover_data={"rank": True, "score": ":.3f"},
                title="Countries by Region",
                height=500,
            )

        fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0},
                              geo=dict(showframe=False, showcoastlines=True))
        if compare_countries:
            for c in compare_countries:
                row = df_full[df_full["country"] == c]
                if not row.empty:
                    fig_map.add_annotation(text=f"★ {c}", showarrow=False,
                                           font=dict(size=10, color="white"))
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.markdown("#### Peace Tier Breakdown")
        tier_counts = df["peaceTier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        tier_order = ["🟢 Very High Peace","🟡 High Peace","🟠 Medium Peace","🔴 Low Peace","⛔ Very Low Peace"]
        tier_counts["tier"] = pd.Categorical(tier_counts["tier"], categories=tier_order, ordered=True)
        tier_counts = tier_counts.sort_values("tier")
        tier_color_map = {
            "🟢 Very High Peace": "#22c55e",
            "🟡 High Peace": "#a3e635",
            "🟠 Medium Peace": "#facc15",
            "🔴 Low Peace": "#f97316",
            "⛔ Very Low Peace": "#ef4444",
        }
        fig_pie = px.pie(tier_counts, names="tier", values="count",
                         color="tier", color_discrete_map=tier_color_map,
                         height=320)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Quick Stats")
        st.metric("Avg Score (filtered)", f"{df['score'].mean():.3f}" if len(df) else "—")
        st.metric("Best Rank in Filter", f"#{int(df['rank'].min())}" if len(df) else "—")
        st.metric("Worst Rank in Filter", f"#{int(df['rank'].max())}" if len(df) else "—")
        improved_n = len(df[df["changeDirection"] == "Improved"])
        st.metric("Improved (filtered)", f"{improved_n} / {len(df)}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Rankings
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"Top {top_n} Most Peaceful")
        top_df = df.sort_values("score").head(top_n).copy()
        top_df["highlight"] = top_df["country"].isin(compare_countries)
        colors = ["#f59e0b" if h else "#22c55e" for h in top_df["highlight"]]
        fig_top = go.Figure(go.Bar(
            y=top_df["country"], x=top_df["score"],
            orientation="h", marker_color=colors,
            text=top_df["score"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            customdata=top_df[["rank","yearChange"]].values,
            hovertemplate="<b>%{y}</b><br>Rank: %{customdata[0]}<br>Score: %{x:.3f}<br>Change: %{customdata[1]:+.3f}<extra></extra>",
        ))
        fig_top.update_layout(height=max(350, top_n * 22), xaxis_title="GPI Score",
                              yaxis=dict(autorange="reversed"), margin=dict(l=10, r=60, t=10, b=30))
        st.plotly_chart(fig_top, use_container_width=True)

    with c2:
        st.subheader(f"Top {top_n} Least Peaceful")
        bot_df = df.sort_values("score", ascending=False).head(top_n).copy()
        bot_df["highlight"] = bot_df["country"].isin(compare_countries)
        colors2 = ["#f59e0b" if h else "#ef4444" for h in bot_df["highlight"]]
        fig_bot = go.Figure(go.Bar(
            y=bot_df["country"], x=bot_df["score"],
            orientation="h", marker_color=colors2,
            text=bot_df["score"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            customdata=bot_df[["rank","yearChange"]].values,
            hovertemplate="<b>%{y}</b><br>Rank: %{customdata[0]}<br>Score: %{x:.3f}<br>Change: %{customdata[1]:+.3f}<extra></extra>",
        ))
        fig_bot.update_layout(height=max(350, top_n * 22), xaxis_title="GPI Score",
                              yaxis=dict(autorange="reversed"), margin=dict(l=10, r=60, t=10, b=30))
        st.plotly_chart(fig_bot, use_container_width=True)

    st.markdown("---")
    st.subheader("Biggest Movers (Year-over-Year)")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("##### 📈 Most Improved")
        imp_df = pd.DataFrame(TOP_IMPROVERS)
        fig_imp = go.Figure(go.Bar(
            y=imp_df["country"], x=imp_df["change"].abs(),
            orientation="h", marker_color="#22c55e",
            text=imp_df["change"].apply(lambda x: f"{x:+.3f}"),
            textposition="outside",
        ))
        fig_imp.update_layout(height=280, xaxis_title="Score Improvement",
                              yaxis=dict(autorange="reversed"), margin=dict(t=5,b=30,l=5,r=60))
        st.plotly_chart(fig_imp, use_container_width=True)

    with c4:
        st.markdown("##### 📉 Most Deteriorated")
        det_df = pd.DataFrame(TOP_DETERIORATORS)
        fig_det = go.Figure(go.Bar(
            y=det_df["country"], x=det_df["change"],
            orientation="h", marker_color="#ef4444",
            text=det_df["change"].apply(lambda x: f"+{x:.3f}"),
            textposition="outside",
        ))
        fig_det.update_layout(height=280, xaxis_title="Score Deterioration",
                              yaxis=dict(autorange="reversed"), margin=dict(t=5,b=30,l=5,r=60))
        st.plotly_chart(fig_det, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Domain Scatter — {domain_focus}")
    domain_col = {"Safety & Security": "safetyScore", "Ongoing Conflict": "conflictScore",
                  "Militarisation": "militarisationScore"}[domain_focus]

    scatter_df = df.copy()
    scatter_df["is_highlighted"] = scatter_df["country"].isin(compare_countries)
    fig_scatter = px.scatter(
        scatter_df, x="score", y=domain_col, color="region",
        size="changeMagnitude", size_max=18,
        hover_name="country",
        hover_data={"rank": True, "score": ":.3f", domain_col: ":.3f", "yearChange": ":.3f", "region": False},
        labels={"score": "Overall GPI Score", domain_col: f"{domain_focus} Score"},
        height=420,
    )
    if compare_countries:
        hl = scatter_df[scatter_df["is_highlighted"]]
        fig_scatter.add_trace(go.Scatter(
            x=hl["score"], y=hl[domain_col], mode="markers+text",
            text=hl["country"], textposition="top center",
            marker=dict(size=16, color="gold", symbol="star", line=dict(width=2, color="black")),
            name="Highlighted", showlegend=True,
        ))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Global Peace Trend (2008–2025)")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend_df["year"], y=trend_df["avgScore"],
            mode="lines+markers", name="Avg GPI Score",
            line=dict(color="#3b82f6", width=3),
            marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
            hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
        ))
        fig_trend.add_hline(y=trend_df["avgScore"].iloc[0], line_dash="dash",
                            line_color="gray", annotation_text="2008 baseline")
        fig_trend.update_layout(height=360, xaxis_title="Year", yaxis_title="Avg GPI Score",
                                hovermode="x unified", yaxis=dict(range=[1.95, 2.15]))
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        st.subheader("Domain Trends (2008–2025)")
        fig_domain = go.Figure()
        domain_colors = {"Safety & Security":"#3b82f6","Ongoing Conflict":"#a855f7","Militarisation":"#22c55e"}
        for col, label, color in [("safety","Safety & Security","#3b82f6"),
                                   ("conflict","Ongoing Conflict","#a855f7"),
                                   ("militarisation","Militarisation","#22c55e")]:
            fig_domain.add_trace(go.Scatter(
                x=domain_df["year"], y=domain_df[col],
                mode="lines", name=label, line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
            ))
        fig_domain.update_layout(height=360, xaxis_title="Year", yaxis_title="Domain Score",
                                 hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_domain, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Economic Cost of Violence ($T)")
        fig_econ = go.Figure()
        fig_econ.add_trace(go.Bar(
            x=econ_df["year"], y=econ_df["trillionsUSD"],
            marker_color="#f97316", name="Cost ($T)",
            hovertemplate="<b>%{x}</b><br>$%{y:.2f}T<extra></extra>",
        ))
        fig_econ.add_trace(go.Scatter(
            x=econ_df["year"], y=econ_df["percentGWP"],
            mode="lines+markers", name="% of GWP", yaxis="y2",
            line=dict(color="#3b82f6", width=2),
        ))
        fig_econ.update_layout(
            height=320,
            yaxis=dict(title="USD Trillions"),
            yaxis2=dict(title="% of GWP", overlaying="y", side="right", range=[10, 13]),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_econ, use_container_width=True)

    with c4:
        st.subheader("2025 Domain Snapshot")
        domains_2025 = domain_df[domain_df["year"] == 2025].iloc[0]
        domains_2008 = domain_df[domain_df["year"] == 2008].iloc[0]
        cats  = ["Safety & Security", "Ongoing Conflict", "Militarisation"]
        v2025 = [domains_2025["safety"], domains_2025["conflict"], domains_2025["militarisation"]]
        v2008 = [domains_2008["safety"], domains_2008["conflict"], domains_2008["militarisation"]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=v2008+[v2008[0]], theta=cats+[cats[0]],
                                            fill="toself", name="2008", line_color="#22c55e"))
        fig_radar.add_trace(go.Scatterpolar(r=v2025+[v2025[0]], theta=cats+[cats[0]],
                                            fill="toself", name="2025", line_color="#ef4444",
                                            fillcolor="rgba(239,68,68,0.15)"))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[1.5, 2.8])),
                                height=320, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Regions
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Regional Peace Scorecard")
    c1, c2 = st.columns([3, 2])

    with c1:
        fig_reg = px.bar(
            region_df.sort_values("avgScore"),
            x="avgScore", y="region", orientation="h",
            color="avgScore", color_continuous_scale="RdYlGn_r",
            range_color=[1.0, 3.2],
            text="avgScore",
            hover_data={"countriesCount": True, "improved": True, "deteriorated": True, "trend": ":.3f"},
            labels={"avgScore": "Avg GPI Score", "region": "Region"},
            height=400,
        )
        fig_reg.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_reg.update_layout(coloraxis_showscale=False, margin=dict(r=60, t=10))
        st.plotly_chart(fig_reg, use_container_width=True)

    with c2:
        st.markdown("#### Improved vs Deteriorated by Region")
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(
            y=region_df["region"], x=region_df["improved"],
            orientation="h", name="Improved", marker_color="#22c55e",
        ))
        fig_div.add_trace(go.Bar(
            y=region_df["region"], x=[-v for v in region_df["deteriorated"]],
            orientation="h", name="Deteriorated", marker_color="#ef4444",
        ))
        fig_div.update_layout(barmode="relative", height=400,
                              xaxis_title="← Deteriorated | Improved →",
                              legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_div, use_container_width=True)

    st.markdown("---")
    st.subheader("Region Detail (filtered countries)")
    reg_detail = df.groupby("region").agg(
        Countries=("country","count"),
        Avg_Score=("score","mean"),
        Best_Score=("score","min"),
        Worst_Score=("score","max"),
        Improved=("changeDirection", lambda x: (x=="Improved").sum()),
        Deteriorated=("changeDirection", lambda x: (x=="Deteriorated").sum()),
    ).reset_index().sort_values("Avg_Score")
    reg_detail["Avg_Score"] = reg_detail["Avg_Score"].round(3)
    reg_detail["Best_Score"] = reg_detail["Best_Score"].round(3)
    reg_detail["Worst_Score"] = reg_detail["Worst_Score"].round(3)
    st.dataframe(reg_detail, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Region-level Treemap")
    treemap_df = df.groupby(["region","peaceTier"]).agg(
        count=("country","count"),
        avg_score=("score","mean"),
    ).reset_index()
    fig_tree = px.treemap(
        treemap_df, path=["region","peaceTier"], values="count",
        color="avg_score", color_continuous_scale="RdYlGn_r",
        range_color=[1.0, 3.5],
        hover_data={"avg_score": ":.3f"},
        height=420,
    )
    fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig_tree, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Indicator Change Analysis (2024 → 2025)")
    c1, c2 = st.columns([3, 2])

    with c1:
        ind_df_sorted = ind_df.sort_values("magnitude", ascending=True)
        colors_ind = ["#22c55e" if d == "improved" else ("#ef4444" if d == "deteriorated" else "#94a3b8")
                      for d in ind_df_sorted["direction"]]
        fig_ind = go.Figure(go.Bar(
            y=ind_df_sorted["name"],
            x=[m if d == "deteriorated" else -m
               for m, d in zip(ind_df_sorted["magnitude"], ind_df_sorted["direction"])],
            orientation="h",
            marker_color=colors_ind,
            hovertemplate="<b>%{y}</b><br>Change: %{x:.3f}<extra></extra>",
        ))
        fig_ind.add_vline(x=0, line_color="white", line_width=1)
        fig_ind.update_layout(height=480, xaxis_title="← Improved | Deteriorated →",
                              margin=dict(l=10, r=10, t=10, b=30))
        st.plotly_chart(fig_ind, use_container_width=True)

    with c2:
        st.markdown("#### By Domain")
        dom_summary = ind_df.groupby(["domain","direction"])["magnitude"].sum().reset_index()
        fig_dom = px.bar(dom_summary, x="magnitude", y="domain", color="direction",
                         color_discrete_map={"improved":"#22c55e","deteriorated":"#ef4444","no change":"#94a3b8"},
                         orientation="h", height=280, barmode="group")
        fig_dom.update_layout(margin=dict(t=10,b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_dom, use_container_width=True)

        st.markdown("#### Conflict Stats 2025")
        st.metric("Active Conflicts", "59", delta="Most since WWII", delta_color="inverse")
        st.metric("Countries in Conflict", "92 / 163")
        st.metric("Conflicts > 1,000 Deaths", "17")
        st.metric("Countries Engaged Abroad", "78")
        st.metric("Conflict Resolution Rate", "4%", delta="Historical low", delta_color="inverse")

    if compare_countries:
        st.markdown("---")
        st.subheader("📊 Country Comparison")
        comp_df = df_full[df_full["country"].isin(compare_countries)].copy()
        if not comp_df.empty:
            domains = ["safetyScore","conflictScore","militarisationScore"]
            domain_labels = ["Safety","Conflict","Militarisation"]
            fig_comp = go.Figure()
            for _, row in comp_df.iterrows():
                fig_comp.add_trace(go.Scatterpolar(
                    r=[row[d] for d in domains] + [row[domains[0]]],
                    theta=domain_labels + [domain_labels[0]],
                    fill="toself", name=row["country"],
                ))
            fig_comp.update_layout(height=380, polar=dict(radialaxis=dict(range=[1.0, 4.0])),
                                   legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_comp, use_container_width=True)

            st.dataframe(
                comp_df[["rank","country","region","score","safetyScore",
                          "conflictScore","militarisationScore","yearChange","peaceTier"]]
                .style.background_gradient(subset=["score"], cmap="RdYlGn_r"),
                use_container_width=True, hide_index=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Data Table
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader(f"Country Data Table — {len(df)} countries")
    c1, c2, c3 = st.columns(3)
    with c1:
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, "gpi_2025_filtered.csv", "text/csv")
    with c2:
        st.metric("Filtered Countries", len(df))
    with c3:
        st.metric("Avg GPI Score", f"{df['score'].mean():.3f}" if len(df) else "—")

    display_cols = ["rank","country","region","score","safetyScore",
                    "conflictScore","militarisationScore","yearChange","changeDirection","peaceTier"]
    styled_df = df[display_cols].rename(columns={
        "rank": "Rank", "country": "Country", "region": "Region",
        "score": "GPI Score", "safetyScore": "Safety", "conflictScore": "Conflict",
        "militarisationScore": "Militarisation", "yearChange": "YoY Change",
        "changeDirection": "Direction", "peaceTier": "Tier",
    })
    st.dataframe(
        styled_df.style.background_gradient(subset=["GPI Score","Safety","Conflict","Militarisation"],
                                            cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True, height=500,
    )

    st.markdown("---")
    st.subheader("Score Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(df, x="score", nbins=25, color="peaceTier",
                                color_discrete_sequence=["#22c55e","#a3e635","#facc15","#f97316","#ef4444"],
                                title="Distribution of GPI Scores",
                                labels={"score":"GPI Score","count":"Countries"})
        fig_hist.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        fig_box = px.box(df, x="region", y="score", color="region",
                         title="Score Spread by Region",
                         labels={"score":"GPI Score","region":"Region"})
        fig_box.update_layout(height=300, showlegend=False,
                              xaxis=dict(tickangle=30, tickfont=dict(size=9)))
        st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — AI Assistant
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🤖 AI Peace Analyst — Powered by Gemini")
    st.caption("Ask anything about the Global Peace Index 2025 data, trends, and geopolitics.")

    if gemini_model is None:
        st.error("Gemini API key not configured. Please add GEMINI_API_KEY to your secrets.")
    else:
        # Build context summary for the model
        top5 = df_full.sort_values("score").head(5)["country"].tolist()
        bot5 = df_full.sort_values("score", ascending=False).head(5)["country"].tolist()

        SYSTEM_CONTEXT = f"""You are an expert analyst specialising in global peace, geopolitics, and the Institute for Economics & Peace (IEP) Global Peace Index 2025.

Key facts from the GPI 2025 report:
- 163 countries monitored; 99.7% of world population
- Global average GPI score: 2.104 (scale 1=most peaceful, 5=least peaceful)
- Global score deteriorated by 0.36% — 13th deterioration in 17 years
- 59 active conflicts — highest since WWII; 92 countries in conflict
- Economic cost of violence: $19.97 trillion (11.6% of global GDP)
- 74 countries improved, 87 deteriorated
- 5 most peaceful: {', '.join(top5)}
- 5 least peaceful: {', '.join(bot5)}
- Biggest improver: Peru (-0.087); Biggest deterioration: Sudan (+0.142)
- Ongoing Conflict domain spiked sharply in 2025 (2.183 in 2008 → 2.558 in 2025)
- Militarisation reversed its decade-long improvement in 2023-2025
- MENA region: avg score 2.870 (worst); Western Europe: 1.476 (best)
- South America improved most in 2025 (Peru, Argentina, Honduras, El Salvador, Guatemala)

Current filter context:
- Showing {len(df)} of 163 countries
- Regions: {', '.join(sel_regions) if sel_regions else 'All'}
- Score range: {score_min:.2f} – {score_max:.2f}
- Change direction filter: {change_dir}

Answer concisely and with data-driven insights. Use bullet points when listing facts."""

        # Init chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Suggested prompts
        if not st.session_state.messages:
            st.markdown("**💡 Try asking:**")
            prompts = [
                "Which region improved the most in 2025?",
                "Why did the Ongoing Conflict score spike so sharply?",
                "What drives Sudan's dramatic deterioration?",
                "Compare peace trends in South Asia vs Western Europe",
                "What is the economic cost of violence in 2025 and why does it matter?",
                "Which countries are outliers in their region?",
            ]
            cols = st.columns(3)
            for i, p in enumerate(prompts):
                if cols[i % 3].button(p, key=f"prompt_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": p})

        # Display history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if user_input := st.chat_input("Ask the AI Peace Analyst…"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analysing…"):
                    try:
                        history = [
                            {"role": m["role"], "parts": [m["content"]]}
                            for m in st.session_state.messages[:-1]
                        ]
                        chat = gemini_model.start_chat(history=history)
                        full_prompt = f"{SYSTEM_CONTEXT}\n\nUser question: {user_input}"
                        response = chat.send_message(full_prompt)
                        reply = response.text
                    except Exception as e:
                        reply = f"⚠️ Error contacting Gemini: {e}"

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

        if st.session_state.messages:
            if st.button("🗑️ Clear conversation"):
                st.session_state.messages = []
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("📊 Global Peace Index 2025 | Institute for Economics & Peace | Data visualised with Streamlit & Plotly")
