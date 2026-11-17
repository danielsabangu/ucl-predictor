# UCL Predictor

A machine-learning system that predicts UEFA Champions League match results as **home / draw / away probabilities**. It combines team strength (Elo), form, squad, lineup, injury and transfer data, picks its production model on a strict season-based validation split, and shows each prediction with a confidence score in a Streamlit app.

**Production model:** CatBoost classifier · 290 features · trained on 2,089 UCL matches (2010/11 to 2025/26)

## Results on unseen data

The 2025/26 season (189 matches) was held back as a final test set and never used for tuning.

| Model | Season | Log loss ↓ | Accuracy | Macro F1 |
|---|---|---|---|---|
| Always predict the most common result (baseline) | 2024/25 validation | 1.002 | 50.8% | 0.22 |
| Random forest (best on validation) | 2024/25 validation | 0.915 | 58.7% | 0.45 |
| **CatBoost (production)** | **2025/26 final test** | **0.934** | **59.8%** | **0.45** |

**Log loss** is the main metric, because the goal is trustworthy probabilities, not just picking winners. Lower is better. A model that only knew how often home teams win scores about 1.0.

**Known weakness:** draws. Draw recall on the test season is about 3%: the model almost never predicts a draw. This is common for football models, because draws are rarely the single most likely result. I tested a dedicated draw layer to fix it, and rejected it because it made the overall probabilities worse.

## How it works

```
Collect data  →  Build features  →  Train & compare models  →  Backtest  →  Predict fixtures  →  Streamlit app
```

1. **Collect** (`src/data/`): matches, fixtures, lineups, injuries, squads, players and transfers, from football-data.org, API-Football and public Kaggle datasets. The data is cleaned and merged into one match table.
2. **Features** (`src/features/`): 411 candidate features, cut to 290 for production:
   - **Elo ratings:** team strength before kick-off and the expected result
   - **Form:** last 5 and 10 matches, home form, season points per game, goal difference, clean sheets
   - **Squad:** strength, continuity, lineup strength, player ratings
   - **Context:** injury impact, transfer impact, rest days
3. **Train and select** (`src/ml/`): a baseline, logistic regression, random forest, extra trees, XGBoost and CatBoost were trained, with the best chosen by log loss on a separate validation season. I also tested an ensemble, probability calibration and a draw layer. Each was **rejected because it didn't improve validation results**, and those decisions are saved in `models/metadata/`.
4. **Backtest:** rolling backtests train only on past seasons and test on the next one, so no future information leaks in. Feature ablations remove one feature group at a time (Elo, UCL form, domestic form, squad, transfers) to measure what each group adds. The production model leaves out the 121 transfer features, keeping 290 of the 411.
5. **Predict** (`src/production/`): one command refreshes the data, rebuilds features, scores upcoming fixtures, and adds:
   - a **Poisson scoreline layer**, for the most likely exact scores
   - an **independent confidence score**, which drops when live data is missing (for example, unknown lineups)
6. **App** (`app.py`): a Streamlit dashboard of upcoming fixtures with probabilities, confidence labels and exportable PDF/PNG "receipts".

## Run it

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # macOS/Linux: .venv/bin/pip

# Optional, only for refreshing live data: copy .env.example to .env and add your API keys
#   FOOTBALL_DATA_API_KEY=...   (football-data.org, free tier)
#   API_FOOTBALL_KEY=...        (API-Football)

.venv\Scripts\streamlit run app.py                 # view the latest saved predictions
.venv\Scripts\python -m src.production.run         # full refresh: data → features → predictions
```

The app works without API keys, using the latest snapshot in `data/app/`.

## Project layout

```
app.py, app/            Streamlit dashboard and receipt exports
src/data/               data collection and cleaning
src/features/           Elo, form, squad, injury, transfer and rest-day features
src/ml/                 training, backtests, calibration, ablations, evaluation
src/production/         live pipeline, app data, health checks, release
models/metadata/        saved metrics and every model-selection decision
notebooks/              exploration → feature engineering → model experiments
tests/                  automated tests
```

## What I learned

- **Pick the metric before the model.** I optimised log loss, so the probabilities are calibrated enough to rank how confident each prediction is.
- **Keep a test set you never touch.** The 2025/26 season was only evaluated once, at the end.
- **"No improvement" is a valid result.** The ensemble, calibration and draw layer all sounded good and all lost on validation, so they're documented and left out.
