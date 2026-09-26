<div align="center">

# 📊 IDSS — AI Use Case Prioritization for Culinary MSMEs

**An interactive Intelligent Decision Support System (IDSS) for prioritizing AI pilot
projects, built with Streamlit and Plotly.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://idss-project-2r2f4hewclsu5zmy9zr4j7.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-FF4B4B)
![Plotly](https://img.shields.io/badge/plotly-5.18%2B-3F4F75)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**[🚀 Live Demo](https://idss-project-2r2f4hewclsu5zmy9zr4j7.streamlit.app/)**

</div>

---

## 📖 Overview

This project turns a classic Multi-Criteria Decision Making (MCDM) case study —
prioritizing which AI use case to pilot first among Culinary MSMEs — into a live,
editable decision-support tool. Instead of a static spreadsheet or slide deck, a
manager can adjust stakeholders, criteria weights, and alternative scores directly in
the browser and instantly see how the recommendation changes.

The app computes and cross-validates results from **three independent MCDM methods**
side by side, so the final recommendation isn't just the output of a single formula —
it's a rank that has been checked for agreement across methods.

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧑‍🤝‍🧑 **Editable stakeholder panel** | Add or remove stakeholders, assign each a role, and let them allocate 100 points across criteria — with automatic total validation |
| 🎚️ **Configurable feasibility gate** | Set optional min/max thresholds per criterion; alternatives that violate them are automatically excluded |
| 📥 **Editable decision matrix** | Update raw scores (C1–C6) for each alternative and see every downstream result recalculate instantly |
| 📊 **Three MCDM methods** | Simple Additive Weighting (**SAW**), Technique for Order Preference by Similarity to Ideal Solution (**TOPSIS**), and Weighted Product (**WP**) |
| 🏆 **Cross-method dashboard** | Compares rankings from all three methods side by side and highlights the overall best alternative in a distinct color |
| 📈 **Interactive Plotly visuals** | Ranking bar charts, score decomposition, average-rank comparison, and a radar chart of each alternative's criteria profile |
| ⬇️ **CSV export** | Download the full comparison table for reporting |

## 🖥️ Live App

👉 **[https://idss-project-2r2f4hewclsu5zmy9zr4j7.streamlit.app/](https://idss-project-2r2f4hewclsu5zmy9zr4j7.streamlit.app/)**

No installation needed — open the link and start editing the data directly in your
browser.

## 🧮 Methodology

The app implements three well-established MCDM methods, all operating on the same
weighted decision matrix so their outputs are directly comparable:

<details>
<summary><strong>SAW — Simple Additive Weighting</strong></summary>

Normalizes each criterion (`x / max` for benefit criteria, `min / x` for cost
criteria), then computes a weighted sum:

```
V_i = Σ_j (w_j · r_ij)
```

</details>

<details>
<summary><strong>TOPSIS — Technique for Order Preference by Similarity to Ideal Solution</strong></summary>

Normalizes the matrix using vector normalization, defines a positive ideal solution
(A*) and a negative ideal solution (A'), then ranks alternatives by their relative
closeness to A*:

```
C_i* = S'_i / (S*_i + S'_i)
```

</details>

<details>
<summary><strong>WP — Weighted Product</strong></summary>

Raises each criterion value to the power of its weight (positive exponent for
benefit criteria, negative for cost criteria), then multiplies across criteria:

```
S_i = Π_j (x_ij ^ w_j)        V_i = S_i / Σ_i S_i
```

</details>

## 🗂️ Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # App theme configuration
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Run locally

```bash
git clone https://github.com/MuhZainur/IDSS-Project.git
cd IDSS-Project
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

### Deploy your own copy (Streamlit Community Cloud)

1. Fork or push this repository to your own GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository and branch, set the main file to
   `app.py`, and click **Deploy**.
4. Every subsequent `git push` to the main branch automatically redeploys the app.

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — interactive web app framework
- [Plotly](https://plotly.com/python/) — interactive charts
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling and computation

## 📚 Background

This tool is based on the case study *"AI Use Case Prioritization for Culinary
MSMEs"* (Special Region of Yogyakarta), originally developed as a SAW-based
decision model for the *Intelligent Decision Support System* course at Universitas
Gadjah Mada, and later extended here with TOPSIS and Weighted Product for
cross-method validation.

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
<sub>Built with Streamlit & Plotly · Maintained by <a href="https://github.com/MuhZainur">MuhZainur</a></sub>
</div>
