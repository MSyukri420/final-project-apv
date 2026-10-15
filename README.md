# Corner Chance Planner

**Which attacking corner setups actually create chances?** A data-driven planning tool for football coaches, built on 879 real corners from StatsBomb 360 data.

Pick the delivery, the target zone and where your players stand. The tool shows how often similar real corners won the first ball and created a good chance, alongside a model estimate tested on competitions it never saw.

![Corner Chance Planner: setting up a corner and comparing real corners with the model](notebooks/images/planner-dark.png)

## Key findings

- **A corner is a two-step chain.** Only about 1 in 3 first balls are won, but about 2 in 3 of those become a good chance. Losing the first ball drops that to fewer than 1 in 5.
- **Target zone is a trade-off.** Far-post balls are won most often (76%), but won balls in the middle turn into chances far more often (80% vs 46%). Overall, the middle creates the most good chances.
- **Spare attackers near the penalty spot matter.** With 2 or more spare attackers there, 42% of corners create a good chance, against 28% with none.
- **The opponent's defensive shape makes no clear difference** once competition is accounted for. Counter-attack risk (about 5% of corners) can't be predicted from the setup, so the tool doesn't pretend to.

## Highlights

- **Honest evaluation.** Every model is tested leave-one-competition-out: train on six competitions, predict the seventh. Predicting the first ball reaches AUC 0.68 on unseen competitions, and models are calibrated.
- **Evidence first.** The app leads with real past corners (with a 90% range) and shows the model as a second opinion, labelled by how reliable it is.
- **Rebuilt after finding a flaw in my own work.** The original version (v1) reported scores measured on its training data. Re-tested properly, it was no better than a coin flip, so I redesigned the targets, features and evaluation from scratch. See [Version history](docs/technical.md#version-history).

**Tech stack:** Python, pandas, scikit-learn (logistic regression, bootstrap ensembles), SHAP, FastAPI, HTML/CSS/JavaScript, Jupyter.

## The app

| Planner | Insights | Playbook |
|---|---|---|
| Set up a corner and compare **real corners** (similar past corners, with a 90% range) with **the model** (an estimate for your exact setup). Quick wins suggest single changes that the models agree help and real corners don't contradict. | What actually happened in real corners, step by step: target zone, delivery type, spare attackers, and the factors that make little difference. | Groups of real corners (delivery, target, players in the box), ranked by good chances. Each opens its average setup in the planner. |

<p>
  <img src="notebooks/images/insights-dark.png" alt="Insights tab" width="49%">
  <img src="notebooks/images/playbook-dark.png" alt="Playbook tab" width="49%">
</p>

The [stakeholder notebook](notebooks/corner_chances_overview.ipynb) explains the findings and the tool for non-technical readers, and reads on GitHub without running anything.

## Run it

```bash
pip install -r requirements.txt
python app.py   # then open http://127.0.0.1:7860
```

The trained models are included, so it works straight away. Retraining steps are in [docs/technical.md](docs/technical.md#running-and-retraining).

## Data

[StatsBomb open data](https://github.com/statsbomb/open-data) with 360 freeze frames: 879 corners from 306 matches in 7 competitions. Used under the StatsBomb [user agreement](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).

## Limitations

- **Sample size**: 879 corners, too few for opponent-specific advice yet.
- **Freeze frames**: one snapshot, showing positions but not runs, marking or player identities.
- **Competitions**: all women's data comes from international tournaments.
- **Excluded**: MLS 2023 and the African Cup of Nations 2023.

**Next steps** that would improve it: more corners (more seasons or a club's own matches), tracking data to capture runs and marking, and player identities for matchup-level advice.

Method, evaluation, project layout and version history: [docs/technical.md](docs/technical.md)
