# WakeWise — Wind Turbine Wake-Loss Predictor

A Streamlit web app based on the feature/target workflow in the uploaded notebook.

## Run locally (Windows)

1. Install Python 3.10 or 3.11.
2. Extract this folder.
3. Open a terminal in this folder and run:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

4. The app opens in your browser. Upload the CSV dataset used in your Colab notebook.

## Required data

The uploaded CSV must contain `y2 (% relative to rated power)`. Predictors are picked from these columns when present:

- `V1`, `V2`, `Va`, `D`
- `y1 (% relative to rated power)`
- `out1`, `out2`
- If `time` exists, app derives `hour`, `day_of_week`, `month`, `day_of_year`

Target follows the notebook: `wake_power_loss_pct = 100 - y2 (% relative to rated power)`.

## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and this README to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app**, select the repository and branch, and set **Main file path** to `app.py`.
4. Deploy. Once the app opens, upload your dataset through the sidebar.

## Notes

- The notebook itself does not contain the raw CSV as an embedded dataset, so the app asks you to upload the CSV to train its models.
- Training occurs in the Streamlit session. The app compares Decision Tree, Random Forest and XGBoost on a held-out 20% test split.
- This is a research prototype. Verify feature availability, units and possible target leakage before using results in a paper or operational decisions.
