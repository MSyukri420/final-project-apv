# Corner Chance Planner

A tool for planning attacking corners. Pick the opponent's defensive shape, the delivery and where your 10 outfield players stand, and it shows how often corners like that win the first ball and create a good chance, from real corners and from a model tested on competitions it never saw.

Built on [StatsBomb open data](https://github.com/statsbomb/open-data) with 360 freeze frames: 879 corners from 306 matches in 7 competitions.

## What it measures

A corner is treated as two steps:

1. **First ball won**: an attacker plays the ball first after the delivery, rather than a defender or the keeper.
2. **Good chance**: a shot worth at least 0.04 xG while the attacking team keeps the ball, within 60 seconds.

```
good chance = first ball won × good chance after winning it
            + first ball lost × good chance after losing it
```

How well each step can be predicted on competitions the model never saw:

| Step | AUC | Label in the app |
|---|---|---|
| First ball won | 0.68 | Tested |
| Good chance after winning it | 0.68 | Tested |
| Good chance after losing it | 0.58 | Weak |
| Good chance overall | 0.57 | Rough guide |

Shots conceded on the counter can't be predicted from the setup (about 5% of corners whatever the setup), so the planner doesn't show them. Expected goals for and against (APV = xG created − xG conceded) are kept as a secondary analyst view.

## Running it

Python 3.12.

```bash
pip install -r requirements.txt

# full pipeline: load data, build features and labels, train and evaluate (several minutes)
python train.py

# retrain from the saved corner table only, after changing model settings
python train.py --models-only

# start the web app, then open http://127.0.0.1:7860
python app.py
```

To use another port, set `PORT`: `PORT=7862 python app.py` in bash, or `$env:PORT=7862; python app.py` in PowerShell.

The trained models and app files (`models/`, `artefacts/`, `outputs/`) are produced by `train.py` and included in the repository, so `python app.py` works straight away. Retraining needs the StatsBomb data in `Statsbomb/`.

### Stakeholder notebook

`notebooks/corner_chances_overview.ipynb` explains the findings and the tool for non-technical readers. It is saved with all its tables and charts, so it can be read in VS Code, Jupyter or on GitHub without running anything. After retraining, rebuild it with:

```bash
python notebooks/build_overview.py
```

## The app

- **Planner**: set up a corner and see the result from **real corners** (similar past corners, with a 90% range) and from **the model** (an estimate for your exact setup), in two tabs. The headline uses real corners when at least 15 similar ones exist. Quick wins suggest single changes that the models agree help and that real corners don't contradict.
- **Insights**: what actually happened in real corners, organised by the two steps: target zone, delivery type and spare attackers at the penalty spot, plus the factors that make little difference.
- **Playbook**: groups of real corners (delivery, target, players in the box), ranked by good chances. Each opens its average setup in the planner.

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

`outputs/reports/train/` after a full run:

- `model_evaluation_train.csv`: every model against the base rate and a shape-only model, on unseen matches and unseen competitions
- `model_per_competition_train.csv`: results for each held-out competition
- `model_horizon_sensitivity_train.csv`: 30, 60 and 90 second windows
- `cluster_eval_train.csv`: rule-based defensive shapes against k-means
- `first_contact_labels_check.csv`: a sample of first-contact labels with the events behind them, for checking by hand

## Limitations

- **Sample size**: 879 corners. Exact player setups are almost never repeated, and most teams defend only a handful of corners, so opponent-specific advice isn't possible.
- **Freeze frames**: one snapshot at the moment of the corner. It shows positions, not runs or marking, and doesn't identify players.
- **Defensive shape**: makes no clear difference once the competition is accounted for. The best setup is the same against every shape.
- **Competitions**: all women's data comes from international tournaments. Women's corners create more good chances (43% vs 28%), mostly after the first ball is lost.
- **Excluded**: MLS 2023 (its corner freeze frames show too few players) and the African Cup of Nations 2023 (almost no 360 data).

## Version history

- **v2 – Corner Chance Planner (current).** Rebuilt around the two-step chain (first ball won, then a good chance), with an evidence-first web app, a stakeholder notebook, and every model tested on competitions it never saw.
- **v1 – Corner APV Coach** (tag [`v1-original`](https://github.com/MSyukri420/final-project-apv/tree/v1-original)). The original final-year project: a Gradio app scoring corners by APV (chance probability minus shot-conceded probability). Re-tested the v2 way, its models did no better than a coin flip on unseen data; its reported scores had been measured on its own training data.

## Data

Data from StatsBomb open data, used under its [user agreement](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
