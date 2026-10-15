← [Back to README](../README.md)

# Corner Chance Planner: technical details

## How a corner is measured

1. **First ball won**: an attacker plays the ball first after the delivery, rather than a defender or the keeper.
2. **Good chance**: a shot worth at least 0.04 xG while the attacking team keeps the ball, within 60 seconds.

```
good chance = first ball won × good chance after winning it
            + first ball lost × good chance after losing it
```

How well each step is predicted on competitions the model never saw (0.5 = coin flip):

| Step | AUC | Label in the app |
|---|---|---|
| First ball won | 0.68 | Tested |
| Good chance after winning it | 0.68 | Tested |
| Good chance after losing it | 0.58 | Weak |
| Good chance overall | 0.57 | Rough guide |

Expected goals for and against (APV = xG created − xG conceded) are kept as a secondary analyst view.

## Data

[StatsBomb open data](https://github.com/statsbomb/open-data) with 360 freeze frames: 879 corners from 306 matches in 7 competitions (World Cup 2022, Euro 2020 and 2024, Women's World Cup 2023, Women's Euro 2022 and 2025, Bundesliga 2023/24, Ligue 1 2021/22 and 2022/23, La Liga 2020/21). Used under the StatsBomb [user agreement](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).

- **Sample size**: exact setups are almost never repeated, and most teams defend only a handful of corners, so opponent-specific advice isn't possible yet.
- **Competitions**: all women's data comes from international tournaments. Women's corners create more good chances (43% vs 28%), mostly after the first ball is lost.
- **Excluded**: MLS 2023 (its freeze frames show too few players) and the African Cup of Nations 2023 (almost no 360 data).

## Running and retraining

Python 3.12.

```bash
pip install -r requirements.txt

# start the web app, then open http://127.0.0.1:7860
python app.py
```

The trained models and app files (`models/`, `artefacts/`, `outputs/`) are included, so the app works straight away. To use another port: `PORT=7862 python app.py` (bash) or `$env:PORT=7862; python app.py` (PowerShell).

To retrain (needs the StatsBomb data in `Statsbomb/`):

```bash
# full pipeline: load data, build features and labels, train and evaluate (several minutes)
python train.py

# retrain from the saved corner table only, after changing model settings
python train.py --models-only

# rebuild the stakeholder notebook after retraining
python notebooks/build_overview.py
```

## Project layout

```
config.yaml           data paths, competitions, model and app settings
train.py              pipeline: stages 1-8
app.py                FastAPI server for the web app
src/
  io_sb.py            loading StatsBomb files
  preprocess.py       events, corners, 360 frame quality filter
  feat_eng.py         attacker and defender counts per zone, delivery features
  clustering.py       rule-based defensive shapes (k-means kept as a benchmark)
  target_eng.py       first-contact and possession-phase outcome labels
  models.py           models, grouped and leave-one-competition-out evaluation, bootstrap
  apv.py              expected-goals APV
  insights.py         app files: corner outcomes, defaults, explanations
  apv_engine.py       scoring a setup against real situations
  coach_service.py    what the app shows: evidence, estimates, quick wins, playbook, insights
web/                  the front end (HTML, CSS, JavaScript)
notebooks/            stakeholder overview notebook

produced by train.py:
models/               trained models and the bootstrap ensemble, loaded by the app
artefacts/            corner outcomes, defaults and explanation data, loaded by the app
outputs/
  train/              corner table and APV tables, reused by --models-only and the notebook
  model_importance/   SHAP feature importance per model
  reports/train/      evaluation tables and figures
```

## Evaluation reports

[`outputs/reports/train/`](../outputs/reports/train/) after a full run:

- [`model_evaluation_train.csv`](../outputs/reports/train/model_evaluation_train.csv): every model against the base rate and a shape-only model, on unseen matches and unseen competitions
- [`model_per_competition_train.csv`](../outputs/reports/train/model_per_competition_train.csv): results for each held-out competition
- [`model_horizon_sensitivity_train.csv`](../outputs/reports/train/model_horizon_sensitivity_train.csv): 30, 60 and 90 second windows
- [`cluster_eval_train.csv`](../outputs/reports/train/cluster_eval_train.csv): rule-based defensive shapes against k-means
- [`first_contact_labels_check.csv`](../outputs/reports/train/first_contact_labels_check.csv): a sample of first-contact labels with the events behind them, for checking by hand

## Version history

- **v2 – Corner Chance Planner (current).** Rebuilt around the two-step chain (first ball won, then a good chance), with an evidence-first web app, a stakeholder notebook, and every model tested on competitions it never saw.
- **v1 – Corner APV Coach** (tag [`v1-original`](https://github.com/MSyukri420/final-project-apv/tree/v1-original)). The original final-year project: a Gradio app scoring corners by APV (chance probability minus shot-conceded probability). Its reported scores (AUC 0.61 for chances, 0.75 for shots conceded) were measured on its own training data. Re-tested on unseen competitions, it did no better than a coin flip and its probabilities were badly inflated (42% predicted chance of conceding a shot vs a real rate of 5%). Finding this led to the v2 rebuild.
