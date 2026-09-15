# JioSaavn Song Trajectory Intelligence — Ultimate Edition

A Django decision-support product for the Song Future Trajectory Prediction project.

## What is included

- Trained Random Forest hit classifier (`model_artifacts/music_hit_prediction_model.pkl`)
- Future-stream Random Forest regressor (`model_artifacts/future_streams_regressor.pkl`)
- 2,000-song prediction dataset and supporting synthetic data
- Portfolio dashboard with live-demo refresh
- Exposure-vs-hit and current-vs-future forecast scatter views
- Hidden opportunity and potential decliner queues
- Language and genre business intelligence
- Action mix and momentum analysis
- New-song prediction cockpit
- Signed local feature-impact explanation in percentage points
- Momentum signal radar
- Counterfactual improvement-lever analysis
- Per-song intelligence page with explanation + signal radar
- Model evidence page with metrics, feature importance, horizon analysis, confusion matrix and regression metrics
- Django admin

> All music/user/streaming data is synthetic/demo data. The dashboard's live mode is a simulation, not a real JioSaavn streaming feed.

## Mac setup

```bash
cd ~/Desktop
unzip JioSaavn_Song_Trajectory_Django_ULTIMATE.zip
cd JioSaavn_Song_Trajectory_Django_ULTIMATE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py makemigrations predictor
python manage.py migrate
python manage.py seed_demo
python manage.py check
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## If the database was created incorrectly

Only if you see `no such table: predictor_songprediction`:

```bash
rm -f db.sqlite3
python manage.py migrate
python manage.py seed_demo
python manage.py runserver
```

Do not delete `data/` or `model_artifacts/`.

## Prediction architecture

The website does not retrain the model when you seed the database.

```text
Trained model (.pkl) ────────────────> /predict/ -> new prediction

Existing prediction CSV -> seed_demo -> SQLite -> dashboard/dataset/song pages
```

The local explanation is a counterfactual diagnostic: one feature is replaced with its training-data median and the change in predicted probability is reported in percentage points. It should be interpreted as model behavior, not causal effect.

## Model retraining / learning from new data

The project now includes **Train / Update Model** at `/train/`.

This is supervised retraining, not automatic learning from unlabeled predictions. A CSV must contain all 11 model features and either:
- `hit` with values 0/1, or
- `future_streams`, from which the app derives `hit` as the top 20% of future-stream outcomes.

Two strategies are available:
- **Combine (recommended):** keep the existing project training data and add the uploaded labelled examples.
- **Replace:** train only on the uploaded labelled dataset.

After a successful retraining, the new Random Forest replaces the active `.pkl`, the future-stream regressor is retrained when `future_streams` is available, the existing song catalog is rescored, and the previous model is backed up under `model_artifacts/backups/`.

Important: simply entering a song's 11 observation features into `/predict/` does **not** train the model. Those inputs have no known outcome yet. To teach the model, upload historical examples where the eventual outcome is known.
