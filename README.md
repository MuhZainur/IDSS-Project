# IDSS — AI Use Case Prioritization for Culinary MSMEs

An interactive Intelligent Decision Support System (IDSS) built with **Streamlit** and
**Plotly**. Managers can edit stakeholders, criteria weights, and the decision matrix,
then instantly compare rankings from three MCDM methods:

- **SAW** (Simple Additive Weighting)
- **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution)
- **WP** (Weighted Product)

## Features

- Editable stakeholder table (add/remove rows) with automatic 100-point validation per row
- Editable min/max feasibility thresholds per criterion
- Editable decision matrix values (C1-C6) per alternative
- Real-time SAW / TOPSIS / WP calculation with step-by-step breakdown
- Interactive Plotly dashboard comparing all three methods, with the overall best
  alternative highlighted in a distinct color
- CSV export of the comparison results

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy (Streamlit Community Cloud — free)

See the step-by-step guide in the accompanying deployment instructions. In short:

1. Push this folder to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, pick this repo/branch, set the main file to `app.py`, click **Deploy**.

## Project structure

```
.
├── app.py                  # main Streamlit app
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml  # optional theme config
└── README.md
```
