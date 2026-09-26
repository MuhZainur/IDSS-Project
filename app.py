"""
IDSS — AI Use Case Prioritization for Culinary MSMEs
Interactive Streamlit app: managers can edit stakeholders, weights, and the decision
matrix, then see SAW, TOPSIS, and Weighted Product results in real time.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ======================================================================================
# PAGE CONFIG
# ======================================================================================
st.set_page_config(
    page_title="IDSS - AI Use Case Prioritization",
    page_icon="📊",
    layout="wide",
)

HIGHLIGHT_COLOR = "#f97316"  # orange — reserved ONLY for the overall best alternative
METHOD_COLORS = {"SAW": "#3b82f6", "TOPSIS": "#22c55e", "WP": "#a855f7"}

# ======================================================================================
# DEFAULT DATA (Culinary MSMEs case study, DIY — same as in the notebook)
# ======================================================================================
DEFAULT_CRITERIA = pd.DataFrame(
    [
        {"kode": "C1", "nama": "Business impact", "arah": "Benefit", "batas_min": None, "batas_maks": None},
        {"kode": "C2", "nama": "Data readiness", "arah": "Benefit", "batas_min": None, "batas_maks": None},
        {"kode": "C3", "nama": "Ease of adoption", "arah": "Benefit", "batas_min": None, "batas_maks": None},
        {"kode": "C4", "nama": "Implementation cost (million IDR)", "arah": "Cost", "batas_min": None, "batas_maks": 20.0},
        {"kode": "C5", "nama": "Implementation time (weeks)", "arah": "Cost", "batas_min": None, "batas_maks": 12.0},
        {"kode": "C6", "nama": "Implementation risk", "arah": "Cost", "batas_min": None, "batas_maks": None},
    ]
)

DEFAULT_ALTERNATIVES = pd.DataFrame(
    [
        {"kode": "A1", "nama": "WhatsApp customer assistant", "C1": 4.0, "C2": 4.4, "C3": 4.8, "C4": 8, "C5": 5, "C6": 2.8},
        {"kode": "A2", "nama": "Demand & inventory forecasting", "C1": 4.8, "C2": 2.8, "C3": 3.1, "C4": 18, "C5": 12, "C6": 3.1},
        {"kode": "A3", "nama": "Automated financial recording", "C1": 4.4, "C2": 4.1, "C3": 4.0, "C4": 10, "C5": 6, "C6": 2.2},
        {"kode": "A4", "nama": "Marketing content generation", "C1": 3.8, "C2": 4.3, "C3": 4.5, "C4": 5, "C5": 3, "C6": 3.2},
        {"kode": "A5", "nama": "AI customer review analysis", "C1": 3.9, "C2": 4.8, "C3": 4.2, "C4": 7, "C5": 4, "C6": 2.6},
    ]
)

DEFAULT_STAKEHOLDERS = pd.DataFrame(
    [
        {"jabatan": "Coordinator", "C1": 30, "C2": 20, "C3": 15, "C4": 10, "C5": 10, "C6": 15},
        {"jabatan": "MSME Owner 1", "C1": 25, "C2": 15, "C3": 20, "C4": 20, "C5": 10, "C6": 10},
        {"jabatan": "MSME Owner 2", "C1": 25, "C2": 20, "C3": 20, "C4": 15, "C5": 10, "C6": 10},
        {"jabatan": "Business mentor", "C1": 30, "C2": 15, "C3": 15, "C4": 15, "C5": 10, "C6": 15},
        {"jabatan": "Technical team", "C1": 20, "C2": 25, "C3": 10, "C4": 15, "C5": 10, "C6": 20},
    ]
)

# ======================================================================================
# SESSION STATE
# ======================================================================================
if "criteria" not in st.session_state:
    st.session_state.criteria = DEFAULT_CRITERIA.copy()
if "alternatives" not in st.session_state:
    st.session_state.alternatives = DEFAULT_ALTERNATIVES.copy()
if "stakeholders" not in st.session_state:
    st.session_state.stakeholders = DEFAULT_STAKEHOLDERS.copy()


def sync_criteria_columns():
    """Keep the alternatives & stakeholders tables' columns in sync with the criteria list."""
    codes = [c for c in st.session_state.criteria["kode"].dropna().tolist() if str(c).strip() != ""]

    alt = st.session_state.alternatives.copy()
    for c in codes:
        if c not in alt.columns:
            alt[c] = 0.0
    keep = ["kode", "nama"] + codes
    alt = alt[[c for c in keep if c in alt.columns]]
    st.session_state.alternatives = alt

    sh = st.session_state.stakeholders.copy()
    for c in codes:
        if c not in sh.columns:
            sh[c] = 0
    keep2 = ["jabatan"] + codes
    sh = sh[[c for c in keep2 if c in sh.columns]]
    st.session_state.stakeholders = sh


def reset_to_default():
    st.session_state.criteria = DEFAULT_CRITERIA.copy()
    st.session_state.alternatives = DEFAULT_ALTERNATIVES.copy()
    st.session_state.stakeholders = DEFAULT_STAKEHOLDERS.copy()


# ======================================================================================
# CORE FUNCTIONS: FEASIBILITY GATE, SAW, TOPSIS, WP
# ======================================================================================
def apply_feasibility(alt_df: pd.DataFrame, criteria_df: pd.DataFrame) -> pd.Series:
    """True/False per alternative based on each criterion's optional batas_min / batas_maks."""
    mask = pd.Series(True, index=alt_df.index)
    for _, row in criteria_df.iterrows():
        code = row["kode"]
        if code not in alt_df.columns:
            continue
        if pd.notna(row.get("batas_min")):
            mask &= alt_df[code] >= row["batas_min"]
        if pd.notna(row.get("batas_maks")):
            mask &= alt_df[code] <= row["batas_maks"]
    return mask


def compute_weights(stakeholder_df: pd.DataFrame, codes: list) -> tuple[pd.Series, pd.Series]:
    """
    First normalize EACH stakeholder's row to sum to 100 (so the tool stays fair even if
    someone mistypes their total), then average across stakeholders, then normalize the
    result to a total weight of 1. Returns (final_weights, raw_point_total_per_stakeholder).
    """
    pts = stakeholder_df[codes].astype(float)
    row_sums = pts.sum(axis=1)
    row_sums_safe = row_sums.replace(0, np.nan)
    normalized_rows = pts.div(row_sums_safe, axis=0) * 100
    mean_points = normalized_rows.mean(axis=0)
    weights = mean_points / mean_points.sum()
    return weights, row_sums


def normalize_saw(matrix: pd.DataFrame, direction: dict) -> pd.DataFrame:
    norm = pd.DataFrame(index=matrix.index, columns=matrix.columns, dtype=float)
    for col in matrix.columns:
        if direction[col] == "Benefit":
            norm[col] = matrix[col] / matrix[col].max()
        else:
            norm[col] = matrix[col].min() / matrix[col]
    return norm


def compute_saw(matrix: pd.DataFrame, weights: pd.Series, direction: dict):
    norm = normalize_saw(matrix, direction)
    weighted = norm * weights[matrix.columns]
    scores = weighted.sum(axis=1)
    return scores, weighted


def compute_topsis(matrix: pd.DataFrame, weights: pd.Series, direction: dict):
    denom = np.sqrt((matrix ** 2).sum(axis=0))
    r = matrix / denom
    v = r * weights[matrix.columns]

    a_pos, a_neg = {}, {}
    for col in matrix.columns:
        if direction[col] == "Benefit":
            a_pos[col], a_neg[col] = v[col].max(), v[col].min()
        else:
            a_pos[col], a_neg[col] = v[col].min(), v[col].max()
    a_pos, a_neg = pd.Series(a_pos), pd.Series(a_neg)

    s_pos = np.sqrt(((v - a_pos) ** 2).sum(axis=1))
    s_neg = np.sqrt(((v - a_neg) ** 2).sum(axis=1))
    closeness = s_neg / (s_pos + s_neg)
    return closeness, v, a_pos, a_neg, s_pos, s_neg


def compute_wp(matrix: pd.DataFrame, weights: pd.Series, direction: dict):
    if (matrix <= 0).any().any():
        return None, None, "Matrix contains values \u2264 0 — WP cannot be computed (raising 0/negative values to a negative power is undefined)."
    signed_w = pd.Series(
        {c: weights[c] if direction[c] == "Benefit" else -weights[c] for c in matrix.columns}
    )
    S = (matrix ** signed_w).prod(axis=1)
    V = S / S.sum()
    return V, S, None


def ranked_bar(series: pd.Series, names: dict, winner: str, x_title: str, title: str) -> go.Figure:
    s = series.sort_values()
    colors = [HIGHLIGHT_COLOR if a == winner else "#93c5fd" for a in s.index]
    labels = [f"{a} — {names.get(a, '')}" for a in s.index]
    fig = go.Figure(
        go.Bar(
            x=s.values, y=labels, orientation="h", marker_color=colors,
            text=[f"{v:.4f}" for v in s.values], textposition="outside",
        )
    )
    fig.update_layout(title=title, xaxis_title=x_title, height=90 + 60 * len(s), margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ======================================================================================
# SIDEBAR
# ======================================================================================
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption(
        "Every value in the tables can be edited by clicking a cell. Criteria & alternative "
        "rows are fixed; only the stakeholder table's rows can be added/removed."
    )
    if st.button("🔄 Reset to example data (Culinary MSMEs)", use_container_width=True):
        reset_to_default()
        st.rerun()
    st.divider()
    st.caption(
        "Built from the *AI Use Case Prioritization for Culinary MSMEs* case study (UGM). "
        "Supports 3 MCDM methods: **SAW**, **TOPSIS**, **Weighted Product**."
    )

st.title("📊 Intelligent Decision Support System (IDSS)")
st.caption("AI Use Case Prioritization — compare SAW, TOPSIS, and Weighted Product interactively")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📋 Criteria & Alternatives", "⚖️ Stakeholder Weights", "📊 SAW", "📐 TOPSIS", "✖️ Weighted Product", "🏆 Comparison Dashboard"]
)

# ======================================================================================
# TAB 1 — CRITERIA & ALTERNATIVES
# ======================================================================================
with tab1:
    st.subheader("1. Criteria List")
    st.caption(
        "Code, name, and direction are fixed and cannot be changed. Only `Min_Threshold` / "
        "`Max_Threshold` (min/max threshold) can be edited — optional, used as a *feasibility gate* "
        "(alternatives that violate the threshold are automatically excluded from the calculation). "
        "Leave blank if there is no threshold."
    )
    edited_criteria = st.data_editor(
        st.session_state.criteria,
        num_rows="fixed",
        use_container_width=True,
        key="criteria_editor",
        column_config={
            "kode": st.column_config.TextColumn("Code", disabled=True),
            "nama": st.column_config.TextColumn("Criterion Name", disabled=True),
            "arah": st.column_config.TextColumn("Direction", disabled=True),
            "batas_min": st.column_config.NumberColumn("Min Threshold (optional)", format="%.2f"),
            "batas_maks": st.column_config.NumberColumn("Max Threshold (optional)", format="%.2f"),
        },
    )
    if not edited_criteria.equals(st.session_state.criteria):
        st.session_state.criteria = edited_criteria
        sync_criteria_columns()
        st.rerun()

    criteria_df = st.session_state.criteria.dropna(subset=["kode"]).copy()
    criteria_df = criteria_df[criteria_df["kode"].astype(str).str.strip() != ""]

    if criteria_df["kode"].duplicated().any():
        st.error("Duplicate criterion codes found — please fix before continuing.")
        st.stop()
    if criteria_df.empty:
        st.error("There must be at least 1 criterion.")
        st.stop()

    criteria_codes = criteria_df["kode"].tolist()
    criteria_names = dict(zip(criteria_df["kode"], criteria_df["nama"]))
    criteria_direction = dict(zip(criteria_df["kode"], criteria_df["arah"]))

    st.subheader("2. Decision Matrix (raw values per alternative)")
    st.caption("Code and use case name are fixed and cannot be changed. Only the C1-C6 values can be edited.")
    edited_alt = st.data_editor(
        st.session_state.alternatives,
        num_rows="fixed",
        use_container_width=True,
        key="alt_editor",
        column_config={
            "kode": st.column_config.TextColumn("Code", disabled=True),
            "nama": st.column_config.TextColumn("Use Case Name", disabled=True),
            **{c: st.column_config.NumberColumn(f"{c}: {criteria_names.get(c, '')}", format="%.2f") for c in criteria_codes},
        },
    )
    st.session_state.alternatives = edited_alt

    alt_df = st.session_state.alternatives.dropna(subset=["kode"]).copy()
    alt_df = alt_df[alt_df["kode"].astype(str).str.strip() != ""]
    if alt_df["kode"].duplicated().any():
        st.error("Duplicate alternative codes found — please fix before continuing.")
        st.stop()
    if len(alt_df) < 2:
        st.error("There must be at least 2 alternatives so they can be compared.")
        st.stop()

    alt_names = dict(zip(alt_df["kode"], alt_df["nama"]))
    matrix_full = alt_df.set_index("kode")[criteria_codes].astype(float)

    st.subheader("3. Feasibility Gate")
    feasible_mask = apply_feasibility(matrix_full, criteria_df)
    gate_view = matrix_full.copy()
    gate_view.insert(0, "name", [alt_names[a] for a in gate_view.index])
    gate_view["Status"] = np.where(feasible_mask, "✅ Pass", "❌ Fail")
    st.dataframe(gate_view, use_container_width=True)
    st.caption(f"{feasible_mask.sum()}/{len(matrix_full)} alternatives pass the gate and are used in the SAW/TOPSIS/WP calculations.")

    matrix = matrix_full.loc[feasible_mask]
    if len(matrix) < 2:
        st.error("Fewer than 2 alternatives pass the feasibility gate — relax batas_min/batas_maks first.")
        st.stop()

# ======================================================================================
# TAB 2 — STAKEHOLDER WEIGHTS
# ======================================================================================
with tab2:
    st.subheader("1. Allocate 100 Points per Stakeholder")
    st.caption(
        "Add/remove rows to add/remove stakeholders. Enter their **Role** and allocate "
        "points across the criteria — each row's total **must equal 100**."
    )
    edited_stakeholders = st.data_editor(
        st.session_state.stakeholders,
        num_rows="dynamic",
        use_container_width=True,
        key="stakeholder_editor",
        column_config={
            "jabatan": st.column_config.TextColumn("Role", required=True),
            **{c: st.column_config.NumberColumn(f"{c}: {criteria_names.get(c, '')}", min_value=0, max_value=100, step=1) for c in criteria_codes},
        },
    )
    st.session_state.stakeholders = edited_stakeholders

    sh_df = st.session_state.stakeholders.dropna(subset=["jabatan"]).copy()
    sh_df = sh_df[sh_df["jabatan"].astype(str).str.strip() != ""]
    if sh_df.empty:
        st.error("There must be at least 1 stakeholder.")
        st.stop()

    weights, row_sums = compute_weights(sh_df, criteria_codes)

    st.subheader("2. Point Total Validation")
    validasi = pd.DataFrame({"Role": sh_df["jabatan"].values, "Total Points": row_sums.values})
    validasi["Status"] = np.where(np.isclose(validasi["Total Points"], 100), "✅ 100", "⚠️ Not 100 (auto-normalized)")
    st.dataframe(validasi, use_container_width=True, hide_index=True)

    bermasalah = validasi[~np.isclose(validasi["Total Points"], 100)]
    if not bermasalah.empty:
        daftar = ", ".join(f"{r.Role} ({r['Total Points']:.0f})" for r in bermasalah.itertuples())
        st.warning(
            f"The following stakeholders' totals are **not 100**: {daftar}. "
            "Their points are still used but have been automatically normalized proportionally "
            "so the calculation doesn't break — it's still best to fix this in the table above for accuracy."
        )
    else:
        st.success("All stakeholders' totals are exactly 100. ✅")

    st.subheader("3. Final Weight per Criterion")
    weights_view = pd.DataFrame(
        {"Criterion": [f"{c}: {criteria_names[c]}" for c in criteria_codes], "Weight": [weights[c] for c in criteria_codes]}
    ).sort_values("Weight", ascending=True)
    fig_w = go.Figure(
        go.Bar(x=weights_view["Weight"], y=weights_view["Criterion"], orientation="h", marker_color="#3b82f6",
               text=[f"{v:.1%}" for v in weights_view["Weight"]], textposition="outside")
    )
    fig_w.update_layout(title="Final Weight per Criterion", xaxis_title="Weight (proportion of 1.0)",
                         height=90 + 45 * len(weights_view), margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_w, use_container_width=True)

# ======================================================================================
# CENTRAL CALCULATION (used by tabs 3-6)
# ======================================================================================
saw_scores, saw_weighted = compute_saw(matrix, weights, criteria_direction)
topsis_scores, topsis_v, topsis_apos, topsis_aneg, topsis_spos, topsis_sneg = compute_topsis(matrix, weights, criteria_direction)
wp_scores, wp_S, wp_error = compute_wp(matrix, weights, criteria_direction)

saw_winner = saw_scores.idxmax()
topsis_winner = topsis_scores.idxmax()
wp_winner = wp_scores.idxmax() if wp_scores is not None else None

# ======================================================================================
# TAB 3 — SAW
# ======================================================================================
with tab3:
    st.subheader("SAW Results (Simple Additive Weighting)")
    st.latex(r"V_i = \sum_j w_j \cdot r_{ij}")

    saw_table = pd.DataFrame({"name": [alt_names[a] for a in saw_scores.index], "SAW Score": saw_scores})
    saw_table = saw_table.sort_values("SAW Score", ascending=False)
    st.dataframe(saw_table, use_container_width=True)
    st.success(f"🏆 SAW Winner: **{saw_winner} — {alt_names[saw_winner]}** (score {saw_scores[saw_winner]:.4f})")

    st.plotly_chart(ranked_bar(saw_scores, alt_names, saw_winner, "SAW Score", "SAW Ranking"), use_container_width=True)

    st.markdown("**Score decomposition per criterion**")
    decomp = saw_weighted.copy()
    decomp.index = [f"{a}: {alt_names[a]}" for a in decomp.index]
    decomp_long = decomp.reset_index().melt(id_vars="index", var_name="Criterion", value_name="Contribution")
    fig_decomp = px.bar(decomp_long, x="Contribution", y="index", color="Criterion", orientation="h",
                         title="Contribution of Each Criterion to the SAW Score")
    fig_decomp.update_layout(yaxis_title="", height=90 + 45 * len(decomp))
    st.plotly_chart(fig_decomp, use_container_width=True)

# ======================================================================================
# TAB 4 — TOPSIS
# ======================================================================================
with tab4:
    st.subheader("TOPSIS Results")
    st.latex(r"C_i^{*} = \frac{S_i'}{S_i^{*} + S_i'}")

    topsis_table = pd.DataFrame(
        {
            "name": [alt_names[a] for a in topsis_scores.index],
            "TOPSIS Score (C*)": topsis_scores,
            "Distance to positive ideal (S*)": topsis_spos,
            "Distance to negative ideal (S')": topsis_sneg,
        }
    ).sort_values("TOPSIS Score (C*)", ascending=False)
    st.dataframe(topsis_table, use_container_width=True)
    st.success(f"🏆 TOPSIS Winner: **{topsis_winner} — {alt_names[topsis_winner]}** (score {topsis_scores[topsis_winner]:.4f})")

    st.plotly_chart(ranked_bar(topsis_scores, alt_names, topsis_winner, "TOPSIS Score (C*)", "TOPSIS Ranking"), use_container_width=True)

    with st.expander("View positive ideal (A*) & negative ideal (A') solutions"):
        st.dataframe(pd.DataFrame({"A* (positive ideal)": topsis_apos, "A' (negative ideal)": topsis_aneg}), use_container_width=True)

# ======================================================================================
# TAB 5 — WEIGHTED PRODUCT
# ======================================================================================
with tab5:
    st.subheader("Weighted Product (WP) Results")
    st.latex(r"S_i = \prod_{j=1}^{n} x_{ij}^{\,w_j} \qquad V_i = \frac{S_i}{\sum_i S_i}")

    if wp_error:
        st.error(wp_error)
    else:
        wp_table = pd.DataFrame({"name": [alt_names[a] for a in wp_scores.index], "WP Score (V)": wp_scores, "S_i": wp_S})
        wp_table = wp_table.sort_values("WP Score (V)", ascending=False)
        st.dataframe(wp_table, use_container_width=True)
        st.success(f"🏆 WP Winner: **{wp_winner} — {alt_names[wp_winner]}** (score {wp_scores[wp_winner]:.4f})")

        st.plotly_chart(ranked_bar(wp_scores, alt_names, wp_winner, "WP Score (V)", "Weighted Product Ranking"), use_container_width=True)

# ======================================================================================
# TAB 6 — COMPARISON DASHBOARD
# ======================================================================================
with tab6:
    st.subheader("SAW vs TOPSIS vs WP Comparison Dashboard")
    st.caption(
        "All three methods happen to produce scores in the 0-1 range, but they mean different "
        "things (weighted-sum aggregate vs. relative closeness to an ideal solution) — comparing "
        "**rankings** across methods is more valid than comparing the raw score magnitudes directly."
    )

    comp = pd.DataFrame({"SAW": saw_scores, "TOPSIS": topsis_scores})
    if wp_scores is not None:
        comp["WP"] = wp_scores
    comp.insert(0, "name", [alt_names[a] for a in comp.index])

    method_cols = [c for c in ["SAW", "TOPSIS", "WP"] if c in comp.columns]
    rank_table = comp[method_cols].rank(ascending=False).astype(int)
    rank_table.columns = [f"Rank_{c}" for c in rank_table.columns]
    comp_full = pd.concat([comp, rank_table], axis=1)

    avg_rank = rank_table.mean(axis=1).sort_values()
    overall_best = avg_rank.index[0]

    winners = {"SAW": saw_winner, "TOPSIS": topsis_winner}
    if wp_winner is not None:
        winners["WP"] = wp_winner
    n_won = sum(1 for w in winners.values() if w == overall_best)

    st.dataframe(comp_full.sort_values("SAW", ascending=False), use_container_width=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.info(
            f"**Winner per method:** " + " | ".join(f"{m}: {w} ({alt_names[w]})" for m, w in winners.items())
        )
    with c2:
        st.success(
            f"🏆 **Overall best:** {overall_best} — {alt_names[overall_best]}\n\n"
            f"Wins directly in {n_won}/{len(method_cols)} methods, average rank = {avg_rank[overall_best]:.2f}"
        )

    # --- Interactive grouped bar chart: score per method, highlighting the overall best ---
    fig = go.Figure()
    for m in method_cols:
        colors = [HIGHLIGHT_COLOR if a == overall_best else METHOD_COLORS[m] for a in comp.index]
        fig.add_trace(
            go.Bar(
                name=m, x=[f"{a}<br>{alt_names[a]}" for a in comp.index], y=comp[m],
                marker_color=colors, text=[f"{v:.3f}" for v in comp[m]], textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group", title="SAW vs TOPSIS vs WP Score per Alternative (orange = overall best)",
        yaxis_title="Score (0-1)", height=500, legend_title="Method",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Average rank ---
    fig_rank = go.Figure(
        go.Bar(
            x=avg_rank.sort_values(ascending=False).values,
            y=[f"{a}: {alt_names[a]}" for a in avg_rank.sort_values(ascending=False).index],
            orientation="h",
            marker_color=[HIGHLIGHT_COLOR if a == overall_best else "#94a3b8" for a in avg_rank.sort_values(ascending=False).index],
            text=[f"{v:.2f}" for v in avg_rank.sort_values(ascending=False).values],
            textposition="outside",
        )
    )
    fig_rank.update_layout(title="Average Rank Across the 3 Methods (1 = best)", xaxis_title="Average rank",
                            height=90 + 45 * len(avg_rank), xaxis_autorange="reversed")
    st.plotly_chart(fig_rank, use_container_width=True)

    # --- Radar chart: normalized criteria profile per alternative ---
    st.markdown("**Criteria profile (SAW-normalized values) — bonus view of each alternative's strengths/weaknesses**")
    norm_saw = normalize_saw(matrix, criteria_direction)
    fig_radar = go.Figure()
    for a in norm_saw.index:
        fig_radar.add_trace(
            go.Scatterpolar(
                r=norm_saw.loc[a, criteria_codes].tolist() + [norm_saw.loc[a, criteria_codes[0]]],
                theta=criteria_codes + [criteria_codes[0]],
                fill="toself",
                name=f"{a}: {alt_names[a]}",
                line_color=HIGHLIGHT_COLOR if a == overall_best else None,
                opacity=1.0 if a == overall_best else 0.55,
            )
        )
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=500,
                             title="Radar Comparison of Criteria Across Alternatives")
    st.plotly_chart(fig_radar, use_container_width=True)

    st.download_button(
        "⬇️ Download comparison results (CSV)",
        comp_full.to_csv(index=True).encode("utf-8"),
        file_name="saw_topsis_wp_comparison.csv",
        mime="text/csv",
    )
