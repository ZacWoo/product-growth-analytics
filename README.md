# ABCflix Subscriber Growth Analytics Dashboard

A polished, interactive Streamlit dashboard for lifecycle-level product growth analytics on a ABCflix-style synthetic subscription dataset.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure the data file is in the same directory as app.py
#    (netflix_synthetic_subscription_data.csv)

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## What This Dashboard Covers

| Section | Key Questions |
|---------|---------------|
| **Overview** | How healthy is the subscriber base? What's the growth trajectory? |
| **Acquisition** | Which channels deliver the highest-quality subscribers (not just volume)? |
| **Retention** | When and why do subscribers churn? Which segments retain best? |
| **Revenue** | Where does revenue concentrate? Are high-volume channels also high-value? |
| **Referral** | Which segments drive organic growth? How do referred users compare? |
| **Recommendations** | What are the 5 highest-impact interventions, grounded in this data? |

## Features

- **Global sidebar filters** across channel, plan, geography, segment, billing interval, and date range
- **Dynamic recommendations** that update based on filter selections
- **Kaplan-Meier–style retention curves** with flexible group-by
- **Cohort retention heatmap** (M0–M12) for monthly signup cohorts
- **Channel quality matrix** (volume vs. churn scatter plot)
- **Revenue decomposition** by tier, channel, and segment with volume-vs-value comparison
- **Referral analysis** with head-to-head comparison of referred vs. non-referred subscribers

## Project Structure

```
netflix-growth-dashboard/
├── app.py          # Main Streamlit application (all 6 sections)
├── helpers.py      # Data loading, feature engineering, utilities
├── requirements.txt
└── README.md
```

## Data

Place `netflix_synthetic_subscription_data.csv` in the same directory as `app.py`. The dataset contains 5,000 synthetic ABCflix-style subscription records with 23 columns covering subscription lifecycle, product engagement, and referral behavior.
