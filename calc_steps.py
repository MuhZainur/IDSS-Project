"""
Display-only "Calculation Process" sections for the SAW, TOPSIS, and WP tabs.

The intermediate steps are recomputed here from the same inputs app.py uses, purely so
they can be shown. The official scores still come from app.py's compute_* functions;
each section compares its final values against them and warns if they ever diverge
(e.g. if the calculation logic in app.py changes).
"""

import numpy as np
import pandas as pd
import streamlit as st

NUM_FMT = "{:.4f}"
NONPOSITIVE_CSS = "background-color: #7f1d1d; color: #fecaca"


# ======================================================================================
# SHARED HELPERS
# ======================================================================================
def _labelled(df: pd.DataFrame, alt_names: dict) -> pd.DataFrame:
    out = df.copy()
    out.index = [f"{a} — {alt_names.get(a, '')}" for a in out.index]
    return out


def _show(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    float_cols = df.select_dtypes("float").columns
    st.dataframe(df.style.format(NUM_FMT, subset=float_cols), use_container_width=True)


def render_step(n: int, title: str, latex: str = None, tables=(), note: str = None):
    """Numbered step: heading, formula, tables (floats shown to 4 decimals), caption."""
    st.markdown(f"**Step {n} — {title}**")
    if latex:
        st.latex(latex)
    for t in tables:
        _show(t)
    if note:
        st.caption(note)


def _criteria_info(weights: pd.Series, direction: dict, cols: list) -> pd.DataFrame:
    return pd.DataFrame(
        {c: [direction[c], f"{weights[c]:.4f}"] for c in cols}, index=["Direction", "Weight (w_j)"]
    )


def _ranking(scores: pd.Series, alt_names: dict, score_col: str) -> pd.DataFrame:
    table = pd.DataFrame({score_col: scores, "Rank": scores.rank(ascending=False, method="min").astype(int)})
    return _labelled(table.sort_values(score_col, ascending=False), alt_names)


def _check_against(step_scores: pd.Series, official: pd.Series, method: str):
    aligned = step_scores.reindex(official.index)
    if len(step_scores) == len(official) and np.allclose(aligned.values, official.values, rtol=1e-9, atol=1e-12):
        st.caption(f"✅ These steps reproduce the {method} scores shown above.")
    else:
        st.warning(
            f"⚠️ The displayed steps don't match the computed {method} scores — the calculation "
            "logic in app.py may have changed. Update calc_steps.py to match."
        )


def _pick_alternative(key: str, index, alt_names: dict) -> str:
    return st.selectbox(
        "Choose an alternative to walk through",
        list(index),
        format_func=lambda a: f"{a} — {alt_names.get(a, '')}",
        key=key,
    )


def _tex(code) -> str:
    return r"\text{" + str(code) + "}"


def _join_lines(terms: list, sep: str, per_line: int = 3) -> str:
    """Join LaTeX terms with `sep`, wrapping every `per_line` terms inside an aligned block."""
    lines = [f" {sep} ".join(terms[i : i + per_line]) for i in range(0, len(terms), per_line)]
    return (r" \\ &\quad " + sep + " ").join(lines)


def _aligned(*rows: str) -> str:
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


# ======================================================================================
# SAW
# ======================================================================================
def show_saw_steps(matrix: pd.DataFrame, weights: pd.Series, direction: dict, alt_names: dict, official_scores: pd.Series):
    with st.expander("🧮 Calculation Process", expanded=False):
        cols = list(matrix.columns)
        w = weights[cols]

        render_step(
            1, "Decision matrix (X)",
            tables=[_labelled(matrix, alt_names), _criteria_info(w, direction, cols)],
            note="Only alternatives that pass the feasibility gate are included.",
        )

        benefit = {c: direction[c] == "Benefit" for c in cols}
        ref = pd.Series({c: matrix[c].max() if benefit[c] else matrix[c].min() for c in cols})
        R = pd.DataFrame({c: matrix[c] / ref[c] if benefit[c] else ref[c] / matrix[c] for c in cols})
        ref_table = pd.DataFrame(
            {c: ["x / max" if benefit[c] else "min / x", f"{ref[c]:g}"] for c in cols},
            index=["Rule", "Reference value"],
        )
        render_step(
            2, "Normalization (R)",
            latex=r"r_{ij} = \frac{x_{ij}}{\max_i x_{ij}} \;\; \text{(benefit)} \qquad "
                  r"r_{ij} = \frac{\min_i x_{ij}}{x_{ij}} \;\; \text{(cost)}",
            tables=[ref_table, _labelled(R, alt_names)],
        )

        weighted = R * w
        render_step(3, "Weighted normalized matrix", latex=r"w_j \cdot r_{ij}", tables=[_labelled(weighted, alt_names)])

        V = weighted.sum(axis=1)
        ranking = _ranking(V, alt_names, "SAW Score (V)")
        render_step(4, "Sum per alternative and rank", latex=r"V_i = \sum_j w_j \cdot r_{ij}", tables=[ranking])
        _check_against(V, official_scores, "SAW")

        st.divider()
        st.markdown("**Worked example**")
        a = _pick_alternative("saw_worked_example", matrix.index, alt_names)
        st.markdown("\n".join(
            f"- {c} ({direction[c]}): r = "
            + (f"{matrix.loc[a, c]:g} / {ref[c]:g}" if benefit[c] else f"{ref[c]:g} / {matrix.loc[a, c]:g}")
            + f" = {R.loc[a, c]:.4f}"
            for c in cols
        ))
        terms = [f"{w[c]:.4f} \\times {R.loc[a, c]:.4f}" for c in cols]
        rank = (V > V[a]).sum() + 1
        st.latex(_aligned(
            f"V_{{{_tex(a)}}} &= \\sum_j w_j \\cdot r_{{{_tex(a)},j}}",
            "&= " + _join_lines(terms, "+"),
            f"&= {V[a]:.4f}",
        ))
        st.caption(f"{a} ranks #{rank} of {len(V)} under SAW.")


# ======================================================================================
# TOPSIS
# ======================================================================================
def show_topsis_steps(matrix: pd.DataFrame, weights: pd.Series, direction: dict, alt_names: dict, official_scores: pd.Series):
    with st.expander("🧮 Calculation Process", expanded=False):
        cols = list(matrix.columns)
        w = weights[cols]

        render_step(
            1, "Decision matrix (X)",
            tables=[_labelled(matrix, alt_names), _criteria_info(w, direction, cols)],
            note="Only alternatives that pass the feasibility gate are included.",
        )

        divisor = np.sqrt((matrix ** 2).sum(axis=0))
        R = matrix / divisor
        render_step(
            2, "Vector normalization (R)",
            latex=r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}",
            tables=[divisor.rename("√Σx²").to_frame().T, _labelled(R, alt_names)],
        )

        Y = R * w
        render_step(3, "Weighted normalized matrix (Y)", latex=r"y_{ij} = w_j \cdot r_{ij}", tables=[_labelled(Y, alt_names)])

        benefit = {c: direction[c] == "Benefit" for c in cols}
        a_pos = pd.Series({c: Y[c].max() if benefit[c] else Y[c].min() for c in cols})
        a_neg = pd.Series({c: Y[c].min() if benefit[c] else Y[c].max() for c in cols})
        render_step(
            4, "Positive ideal (A⁺) and negative ideal (A⁻) solutions",
            latex=r"A^{+}_j = \begin{cases} \max_i y_{ij} & \text{benefit} \\ \min_i y_{ij} & \text{cost} \end{cases} \qquad "
                  r"A^{-}_j = \begin{cases} \min_i y_{ij} & \text{benefit} \\ \max_i y_{ij} & \text{cost} \end{cases}",
            tables=[pd.DataFrame({"A⁺ (positive ideal)": a_pos, "A⁻ (negative ideal)": a_neg}).T],
        )

        d_pos = np.sqrt(((Y - a_pos) ** 2).sum(axis=1))
        d_neg = np.sqrt(((Y - a_neg) ** 2).sum(axis=1))
        render_step(
            5, "Distance to each ideal",
            latex=r"D^{+}_i = \sqrt{\sum_j (y_{ij} - A^{+}_j)^2} \qquad D^{-}_i = \sqrt{\sum_j (y_{ij} - A^{-}_j)^2}",
            tables=[_labelled(pd.DataFrame({"D⁺": d_pos, "D⁻": d_neg}), alt_names)],
        )

        C = d_neg / (d_pos + d_neg)
        ranking = _ranking(C, alt_names, "TOPSIS Score (C*)")
        render_step(6, "Closeness coefficient and rank", latex=r"C^{*}_i = \frac{D^{-}_i}{D^{+}_i + D^{-}_i}", tables=[ranking])
        _check_against(C, official_scores, "TOPSIS")

        st.divider()
        st.markdown("**Worked example**")
        a = _pick_alternative("topsis_worked_example", matrix.index, alt_names)
        st.markdown("\n".join(
            f"- {c}: y = {w[c]:.4f} × {matrix.loc[a, c]:g} / {divisor[c]:.4f} = {Y.loc[a, c]:.4f}"
            for c in cols
        ))
        ta = _tex(a)
        for sign, ideal, dist in (("+", a_pos, d_pos), ("-", a_neg, d_neg)):
            terms = [f"({Y.loc[a, c]:.4f} - {ideal[c]:.4f})^2" for c in cols]
            st.latex(_aligned(
                f"(D^{{{sign}}}_{{{ta}}})^2 &= " + _join_lines(terms, "+"),
                f"&= {dist[a] ** 2:.6f}",
                f"D^{{{sign}}}_{{{ta}}} &= \\sqrt{{{dist[a] ** 2:.6f}}} = {dist[a]:.4f}",
            ))
        rank = (C > C[a]).sum() + 1
        st.latex(
            f"C^{{*}}_{{{ta}}} = \\frac{{{d_neg[a]:.4f}}}{{{d_pos[a]:.4f} + {d_neg[a]:.4f}}} = {C[a]:.4f}"
        )
        st.caption(f"{a} ranks #{rank} of {len(C)} under TOPSIS.")


# ======================================================================================
# WEIGHTED PRODUCT
# ======================================================================================
def show_wp_steps(matrix: pd.DataFrame, weights: pd.Series, direction: dict, alt_names: dict,
                  official_scores: pd.Series, error: str = None):
    with st.expander("🧮 Calculation Process", expanded=False):
        cols = list(matrix.columns)
        w = weights[cols]

        if official_scores is None:
            st.error(error or "WP could not be computed.")
            st.markdown("Values ≤ 0 in the decision matrix are highlighted:")
            view = _labelled(matrix, alt_names)
            st.dataframe(
                view.style.format(NUM_FMT).apply(
                    lambda d: pd.DataFrame(np.where(d <= 0, NONPOSITIVE_CSS, ""), index=d.index, columns=d.columns),
                    axis=None,
                ),
                use_container_width=True,
            )
            return

        render_step(
            1, "Decision matrix (X)",
            tables=[_labelled(matrix, alt_names), _criteria_info(w, direction, cols)],
            note="Only alternatives that pass the feasibility gate are included.",
        )

        exponent = pd.Series({c: w[c] if direction[c] == "Benefit" else -w[c] for c in cols})
        exp_table = pd.DataFrame(
            {c: [direction[c], f"{w[c]:.4f}", f"{exponent[c]:+.4f}"] for c in cols},
            index=["Direction", "Weight (w_j)", "Exponent"],
        )
        render_step(
            2, "Exponents",
            latex=r"e_j = \begin{cases} +w_j & \text{benefit} \\ -w_j & \text{cost} \end{cases}",
            tables=[exp_table],
            note=f"The weights already sum to {w.sum():.4f}, so no further weight normalization is needed.",
        )

        P = matrix ** exponent
        render_step(3, "Raise each value to its exponent", latex=r"x_{ij}^{\,e_j}", tables=[_labelled(P, alt_names)])

        S = P.prod(axis=1)
        render_step(
            4, "Multiply across criteria (S)",
            latex=r"S_i = \prod_j x_{ij}^{\,e_j}",
            tables=[_labelled(S.rename("S_i").to_frame(), alt_names)],
            note=f"ΣS = {S.sum():.4f}",
        )

        V = S / S.sum()
        ranking = _ranking(V, alt_names, "WP Score (V)")
        render_step(5, "Relative preference and rank", latex=r"V_i = \frac{S_i}{\sum_i S_i}", tables=[ranking])
        _check_against(V, official_scores, "WP")

        st.divider()
        st.markdown("**Worked example**")
        a = _pick_alternative("wp_worked_example", matrix.index, alt_names)
        ta = _tex(a)
        powers = [f"{matrix.loc[a, c]:g}^{{{exponent[c]:.4f}}}" for c in cols]
        values = [f"{P.loc[a, c]:.4f}" for c in cols]
        rank = (V > V[a]).sum() + 1
        st.latex(_aligned(
            f"S_{{{ta}}} &= " + _join_lines(powers, r"\times"),
            "&= " + _join_lines(values, r"\times"),
            f"&= {S[a]:.4f}",
        ))
        st.latex(f"V_{{{ta}}} = \\frac{{S_{{{ta}}}}}{{\\sum_i S_i}} = \\frac{{{S[a]:.4f}}}{{{S.sum():.4f}}} = {V[a]:.4f}")
        st.caption(f"{a} ranks #{rank} of {len(V)} under WP.")
