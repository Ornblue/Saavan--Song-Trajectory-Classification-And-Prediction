# Saavan Song Trajectory Intelligence 

A Django-based machine learning decision-support platform for predicting the future trajectory of songs using historical streaming, engagement, exposure, and momentum signals.

The project is designed around a practical business question:

> Given what we know about a song today, what is likely to happen to it in the future?

Instead of treating song popularity as a single static label, this project models a song as something that can grow, stabilize, decline, or become a hidden opportunity. The system combines supervised machine learning, feature engineering, analytics, explainability, and an interactive Django dashboard to turn streaming observations into actionable intelligence.

> **Important:** All music, user, and streaming data included in this project are synthetic/demo data. The dashboard's live mode is a simulation and is not connected to a real JioSaavn streaming feed.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Project Objective](#project-objective)
4. [What Is Song Trajectory?](#what-is-song-trajectory)
5. [Key Features](#key-features)
6. [System Architecture](#system-architecture)
7. [End-to-End Data Flow](#end-to-end-data-flow)
8. [Dataset](#dataset)
9. [Feature Engineering](#feature-engineering)
10. [The 11 Model Features](#the-11-model-features)
11. [Target Variables](#target-variables)
12. [Machine Learning Pipeline](#machine-learning-pipeline)
13. [Hit Classification](#hit-classification)
14. [Future Streams Regression](#future-streams-regression)
15. [Why Random Forest?](#why-random-forest)
16. [Classification vs Regression](#classification-vs-regression)
17. [Prediction Probability](#prediction-probability)
18. [Business Intelligence](#business-intelligence)
19. [Hidden Opportunities](#hidden-opportunities)
20. [Potential Decliners](#potential-decliners)
21. [Momentum Analysis](#momentum-analysis)
22. [Exposure vs Hit Analysis](#exposure-vs-hit-analysis)
23. [Current vs Future Forecast](#current-vs-future-forecast)
24. [Language Intelligence](#language-intelligence)
25. [Genre Intelligence](#genre-intelligence)
26. [Action Mix](#action-mix)
27. [New Song Prediction Cockpit](#new-song-prediction-cockpit)
28. [Model Explainability](#model-explainability)
29. [Counterfactual Analysis](#counterfactual-analysis)
30. [Per-Song Intelligence](#per-song-intelligence)
31. [Model Evidence](#model-evidence)
32. [Model Evaluation Metrics](#model-evaluation-metrics)
33. [Confusion Matrix](#confusion-matrix)
34. [Feature Importance](#feature-importance)
35. [Horizon Analysis](#horizon-analysis)
36. [Django Architecture](#django-architecture)
37. [Database and ORM](#database-and-orm)
38. [Prediction Architecture](#prediction-architecture)
39. [Model Retraining](#model-retraining)
40. [Training Dataset Requirements](#training-dataset-requirements)
41. [Combine vs Replace Training](#combine-vs-replace-training)
42. [Learning From New Data](#learning-from-new-data)
43. [Model Backup and Rescoring](#model-backup-and-rescoring)
44. [Avoiding Data Leakage](#avoiding-data-leakage)
45. [Web Application](#web-application)
46. [Dashboard](#dashboard)
47. [Live Demo Refresh](#live-demo-refresh)
48. [Decision-Support Workflow](#decision-support-workflow)
49. [Django Admin](#django-admin)
50. [Project Structure](#project-structure)
51. [Technology Stack](#technology-stack)
52. [Installation on macOS](#installation-on-macos)
53. [Database Setup](#database-setup)
54. [Running the Application](#running-the-application)
55. [Troubleshooting](#troubleshooting)
56. [Environment Variables](#environment-variables)
57. [Git and GitHub](#git-and-github)
58. [Git Ignore](#git-ignore)
59. [Model Training Philosophy](#model-training-philosophy)
60. [Interpretation Rules](#interpretation-rules)
61. [Limitations](#limitations)
62. [Future Improvements](#future-improvements)
63. [Glossary](#glossary)
64. [Project Summary](#project-summary)

---

# Project Overview

JioSaavn Song Trajectory Intelligence is an end-to-end machine learning web application built with Django.

The application takes song-level observations and uses trained machine learning models to estimate:

- Whether a song is likely to become a hit.
- The probability of the song being classified as a hit.
- The expected future streaming performance.
- Whether the song appears to be an opportunity or a potential decliner.
- Which behavioral signals are influencing the model's prediction.
- What changes in individual input signals could alter the model's prediction.

The project is not simply a machine learning notebook.

It combines:

```text
Data
  ↓
Feature Engineering
  ↓
Machine Learning
  ↓
Predictions
  ↓
Explainability
  ↓
Business Intelligence
  ↓
Django Web Application
  ↓
Decision Support
```

This makes the project closer to a deployable analytics product than a standalone machine learning experiment.

---

# Problem Statement

Music platforms generate enormous amounts of behavioral information around songs.

A song may receive:

- Streams
- Searches
- Saves
- Shares
- Playlist placements
- Exposure
- Engagement
- Skips
- Rapid growth

However, current popularity does not necessarily tell us what will happen next.

A song with moderate current streams may have strong momentum and become a future hit.

Conversely, a song with high current streams may be losing momentum and could decline.

Therefore, the central problem is:

> **How can historical and current song-level signals be used to estimate a song's future trajectory?**

The project addresses this using two complementary machine learning problems:

1. **Classification:** Is the song likely to become a hit?
2. **Regression:** How many future streams is the song expected to generate?

---

# Project Objective

The main objective is to build a decision-support system that transforms song-level behavioral data into useful predictions and recommendations.

The system should allow an analyst or music-business user to:

1. Inspect the current state of a song.
2. Estimate its probability of becoming a hit.
3. Forecast future streams.
4. Identify promising songs that may currently be underexposed.
5. Identify songs whose current popularity may not be sustainable.
6. Understand which signals are driving the prediction.
7. Experiment with hypothetical improvements.
8. Compare groups of songs by language and genre.
9. Evaluate the underlying machine learning models.
10. Retrain the models when new labelled historical data becomes available.

---

# What Is Song Trajectory?

A song trajectory describes how a song's performance changes over time.

A simplified trajectory can be represented as:

```text
                     Future Performance
                           ↑
                           |
             Rising       |       Hit / Breakout
               ↗          |
              /           |
-------------/------------+----------------→ Time
            /             |
           /              |
     Stable              Declining
```

The important idea is that a song is not evaluated only from its current stream count.

Instead, the system considers multiple signals simultaneously.

For example:

```text
Song A
Current streams: Moderate
Growth: High
Momentum: High
Engagement: High
Exposure: Low

Possible interpretation:
Underexposed but gaining traction
```

Another song might have:

```text
Song B
Current streams: High
Growth: Negative
Momentum: Low
Skip rate: High

Possible interpretation:
Currently popular but potentially declining
```

This is why trajectory prediction is more useful than simply ranking songs by their current stream count.

---

# Key Features

The application includes the following major capabilities:

- Trained Random Forest hit classifier.
- Trained Random Forest future-stream regressor.
- Song prediction dataset.
- Synthetic streaming and music data.
- Interactive portfolio dashboard.
- Live-demo refresh simulation.
- Exposure-vs-hit analysis.
- Current-vs-future forecast analysis.
- Hidden opportunity detection.
- Potential decliner detection.
- Language-level business intelligence.
- Genre-level business intelligence.
- Action mix analysis.
- Momentum analysis.
- New-song prediction cockpit.
- Local feature-impact explanations.
- Momentum signal radar.
- Counterfactual improvement-lever analysis.
- Per-song intelligence pages.
- Model evidence and evaluation pages.
- Django administration.
- Supervised model retraining.
- Combine or replace training strategies.
- Automatic backup of the previous model.
- Rescoring of the existing song catalog after retraining.

---

# System Architecture

The complete architecture can be represented as:

```text
                  ┌───────────────────────┐
                  │   Historical Dataset  │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Feature Engineering   │
                  └───────────┬───────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Machine Learning Layer    │
              │                               │
              │ Random Forest Classifier      │
              │ Random Forest Regressor       │
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Prediction Layer │
                    └────────┬─────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
       ┌───────────┐   ┌────────────┐   ┌─────────────┐
       │ Hit Score │   │ Future     │   │ Explanation │
       │           │   │ Streams    │   │             │
       └─────┬─────┘   └─────┬──────┘   └──────┬──────┘
             │               │                 │
             └───────────────┼─────────────────┘
                             ▼
                   ┌───────────────────┐
                   │ Django Application│
                   └─────────┬─────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Analytics Dashboard  │
                  └──────────────────────┘
```

---

# End-to-End Data Flow

The application's complete flow is:

```text
Historical song data
        ↓
Cleaning and feature preparation
        ↓
Training dataset
        ↓
Random Forest models
        ↓
Serialized model artifacts (.pkl)
        ↓
Django prediction layer
        ↓
Song-level predictions
        ↓
SQLite database
        ↓
Dashboard / analytics / song pages
        ↓
Business decisions
```

For a new prediction:

```text
User enters 11 song features
        ↓
Django validates input
        ↓
Features are arranged in model order
        ↓
Trained model receives the input
        ↓
Hit probability is generated
        ↓
Future streams are estimated
        ↓
Explanation is calculated
        ↓
Result is displayed
```

---

# Dataset

The project contains synthetic/demo datasets representing song-level behavior.

The data is intentionally synthetic so that the project can be distributed and demonstrated without exposing real user or streaming information.

The dataset conceptually represents:

- Song characteristics.
- Streaming activity.
- Engagement.
- Exposure.
- Discovery behavior.
- Momentum.
- Future performance.

The project also includes a prediction dataset that powers the dashboard and song intelligence pages.

The training system supports labelled historical data for supervised learning.

---

# Feature Engineering

Feature engineering converts raw behavioral information into numerical signals that machine learning algorithms can use.

Instead of asking the model to understand an entire streaming history directly, the project summarizes song behavior into meaningful features.

Examples include:

```text
Current streaming activity
        +
Growth behavior
        +
Momentum
        +
Audience engagement
        +
Exposure
        +
Discovery behavior
        ↓
Model-ready feature vector
```

A feature vector can be represented as:

```text
X =
[
    streams,
    daily_streams,
    growth_rate,
    momentum,
    exposure,
    engagement,
    skip_rate,
    save_rate,
    share_rate,
    playlist_rate,
    search_rate
]
```

---

# The 11 Model Features

The project uses 11 core model features.

## 1. Streams

Represents the song's observed streaming activity.

It provides a measure of current popularity.

---

## 2. Daily Streams

Represents the recent daily streaming level.

This is useful because two songs can have similar total streams while having very different current activity.

---

## 3. Growth Rate

Represents the rate at which the song's streaming performance is changing.

A positive growth rate can indicate increasing interest.

A negative growth rate can indicate declining interest.

---

## 4. Momentum

Momentum represents the strength and direction of recent movement.

A song with high momentum may be accelerating even if its absolute stream count is not yet large.

---

## 5. Exposure

Exposure represents how much visibility the song receives through platform mechanisms such as playlists or other discovery surfaces represented in the dataset.

Exposure is important because popularity can depend on how much opportunity a song receives to reach listeners.

---

## 6. Engagement

Engagement summarizes how actively listeners interact with the song.

Higher engagement can provide a stronger signal than streams alone.

---

## 7. Skip Rate

Skip rate represents the proportion of listening behavior associated with skipping.

Higher skip behavior can be an unfavorable signal.

---

## 8. Save Rate

Save rate represents the tendency of listeners to save the song.

Saving can indicate stronger listener intent or preference.

---

## 9. Share Rate

Share rate represents how frequently listeners share the song.

Sharing can indicate stronger audience enthusiasm.

---

## 10. Playlist Rate

Playlist rate represents the degree to which playlist activity contributes to the song's exposure.

Playlists can be an important discovery mechanism.

---

## 11. Search Rate

Search rate represents the degree of active discovery behavior around the song.

A high search signal may indicate that listeners are actively looking for the song.

---

# Target Variables

The project uses two machine learning targets.

## Target 1: Hit

The hit target is a binary classification target.

```text
hit = 0
```

means the song is not classified as a hit.

```text
hit = 1
```

means the song is classified as a hit.

The training system can also derive the hit label from `future_streams` by treating the top 20% of future-stream outcomes as hits.

---

## Target 2: Future Streams

Future streams are a continuous numerical target.

Unlike hit classification, this target does not answer a yes/no question.

Instead, it estimates the future streaming outcome.

Conceptually:

```text
Current observations
        ↓
Regression model
        ↓
Estimated future streams
```

---

# Machine Learning Pipeline

The machine learning pipeline is:

```text
Raw labelled data
        ↓
Validation
        ↓
Feature selection
        ↓
Train/test split
        ↓
Model training
        ↓
Evaluation
        ↓
Model serialization
        ↓
Django inference
```

Two models are trained:

```text
                  Input Features
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Random Forest          Random Forest
       Classifier             Regressor
             │                   │
             ▼                   ▼
       Hit Probability      Future Streams
```

---

# Hit Classification

The hit classifier answers:

> How likely is this song to be a hit?

The model is a Random Forest classifier.

The classifier produces:

```text
Class = 0 or 1
```

and a probability:

```text
P(hit)
```

For example:

```text
Hit probability = 0.82
```

means the model estimates an 82% probability for the hit class.

This should be interpreted as a model prediction, not a guarantee of future success.

---

# Future Streams Regression

The regression model answers:

> What future streaming level does the model expect?

Unlike classification, regression predicts a continuous value.

For example:

```text
Predicted future streams
= 1,850,000
```

The exact prediction depends on the input features and trained model.

The regressor is also implemented using Random Forest.

---

# Why Random Forest?

Random Forest is an ensemble machine learning algorithm.

Instead of relying on one decision tree, it builds multiple decision trees and combines their predictions.

Conceptually:

```text
                 Training Data
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
      Tree 1        Tree 2        Tree 3
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                Combined Output
```

Random Forest is useful for this project because:

- It can model nonlinear relationships.
- It can capture interactions between features.
- It works well with tabular data.
- It requires relatively little preprocessing.
- It can provide feature-importance information.
- It supports both classification and regression.

For this type of structured song-level dataset, it is a practical baseline and production-style model choice.

---

# Classification vs Regression

The project deliberately uses both machine learning paradigms.

## Classification

Classification predicts categories.

```text
Input
 ↓
Classifier
 ↓
Hit / Not Hit
```

Example:

```text
hit = 1
```

---

## Regression

Regression predicts a numerical quantity.

```text
Input
 ↓
Regressor
 ↓
Future Streams
```

Example:

```text
future_streams = 1,850,000
```

Together, they provide a richer view:

```text
Will it become a hit?
        +
How much future activity might it generate?
```

---

# Prediction Probability

The classifier provides a probability rather than only a hard label.

This enables ranking.

For example:

```text
Song A → 0.91
Song B → 0.76
Song C → 0.53
Song D → 0.18
```

An analyst can therefore prioritize songs based on estimated potential rather than treating all positive predictions equally.

---

# Business Intelligence

Machine learning predictions become much more useful when combined with business analytics.

The dashboard therefore provides several decision-oriented views.

The general philosophy is:

```text
Machine Learning
      ↓
Prediction
      ↓
Context
      ↓
Prioritization
      ↓
Action
```

The project does not stop at:

> "The model says this song is a hit."

It attempts to answer:

> "Which songs deserve attention, why, and what should be investigated?"

---

# Hidden Opportunities

A hidden opportunity is a song that appears to have potential but may not currently have proportional visibility.

A conceptual example:

```text
Current streams: Moderate
Momentum: High
Engagement: High
Exposure: Low
Hit probability: High
```

This combination can identify songs worth investigating for additional promotion or exposure.

The queue is therefore designed as a decision-support mechanism rather than a strict business rule.

---

# Potential Decliners

The application also identifies songs whose current performance may be vulnerable.

A conceptual pattern is:

```text
Current streams: High
Growth: Negative
Momentum: Weak
Skip rate: High
Future forecast: Lower
```

Such songs may deserve monitoring even though their current popularity remains high.

---

# Momentum Analysis

Momentum is one of the most important concepts in trajectory prediction.

A song can have:

```text
High streams + declining momentum
```

or:

```text
Moderate streams + accelerating momentum
```

These two songs should not necessarily receive the same strategic treatment.

The dashboard therefore exposes momentum-related analytics separately from absolute popularity.

---

# Exposure vs Hit Analysis

The exposure-vs-hit view compares visibility with predicted hit status.

Conceptually:

```text
             Hit Probability
                   ↑
                   |
        Opportunity|    Strong performers
                   |
-------------------+--------------------→ Exposure
                   |
        Low signal |    Highly exposed
                   |
```

This helps investigate whether a song's predicted potential is aligned with the amount of exposure it receives.

---

# Current vs Future Forecast

Current performance and future performance are shown together.

The purpose is to distinguish:

```text
Currently strong
```

from:

```text
Expected to become stronger
```

and:

```text
Currently strong
but expected to decline
```

This makes the dashboard more trajectory-oriented than a simple popularity ranking.

---

# Language Intelligence

The application groups song-level intelligence by language.

This can be used to compare:

- Number of songs.
- Average predicted performance.
- Hit probability.
- Momentum.
- Future-stream forecasts.

The goal is not merely descriptive reporting.

Language-level aggregation can help identify broader patterns in the dataset.

---

# Genre Intelligence

Genre-level analytics provide another aggregation layer.

The dashboard can compare genres using model predictions and behavioral indicators.

A simplified view:

```text
Genre
  ↓
Songs
  ↓
Behavioral Features
  ↓
Model Predictions
  ↓
Aggregated Intelligence
```

---

# Action Mix

The action mix summarizes the types of recommended or implied actions across the song portfolio.

This turns individual predictions into an overall portfolio-level picture.

For example, the portfolio can contain different groups such as:

```text
Promote
Monitor
Maintain
Investigate
```

The exact action interpretation depends on the application's decision-support logic.

---

# New Song Prediction Cockpit

The new-song prediction interface allows users to enter the 11 model features manually.

The flow is:

```text
User Input
    ↓
11 Features
    ↓
Validation
    ↓
Classifier
    ↓
Hit Probability
    ↓
Regressor
    ↓
Future Streams
    ↓
Explanation
```

This makes the trained model usable outside the original dataset.

Important:

> Entering a new song into the prediction cockpit does not automatically train the model.

The song does not yet have a known outcome.

It is an inference request, not a supervised training example.

---

# Model Explainability

Machine learning predictions are more useful when users can understand why the model produced them.

The application therefore provides local diagnostic explanations.

The explanation layer includes:

- Feature impact.
- Signed contribution.
- Momentum signal radar.
- Counterfactual sensitivity.
- Improvement-lever analysis.
- Per-song intelligence.

The purpose is to answer:

> What signals are pushing this prediction upward or downward?

---

# Feature Impact

Feature impact represents how individual input variables influence the local model prediction.

A signed explanation can be expressed conceptually as:

```text
Feature              Impact
--------------------------------
Momentum             +12.4 pp
Engagement            +8.1 pp
Exposure              -3.2 pp
Skip Rate             -6.7 pp
```

Here:

```text
pp = percentage points
```

A positive value indicates that the feature pushes the model's prediction upward relative to the chosen local baseline.

A negative value indicates a downward influence.

These values describe model behavior.

They should not automatically be interpreted as causal effects.

---

# Counterfactual Analysis

The application provides a counterfactual diagnostic.

The system takes one feature at a time and replaces its observed value with the training-data median.

It then compares the new prediction with the original prediction.

Conceptually:

```text
Original song
     ↓
Prediction A

Change one feature to reference value
     ↓
Prediction B

Difference
     ↓
Estimated model sensitivity
```

For example:

```text
Original hit probability = 72%

Momentum replaced with median
New probability = 61%

Difference = -11 percentage points
```

This means that, under this diagnostic, changing momentum to the reference value changed the model output by 11 percentage points.

It does **not** prove that changing momentum in the real world would cause an 11-point improvement or decline.

---

# Per-Song Intelligence

Each song can have its own intelligence page.

A typical song-level view combines:

```text
Song Information
      +
Current Signals
      +
Hit Probability
      +
Future Forecast
      +
Feature Impact
      +
Momentum Radar
      +
Counterfactual Analysis
```

This provides a complete profile instead of forcing users to inspect individual charts separately.

---

# Model Evidence

The model evidence page exposes the underlying machine learning evidence.

It includes:

- Model metrics.
- Feature importance.
- Horizon analysis.
- Confusion matrix.
- Regression metrics.

This is useful for transparency and portfolio demonstration because the application does not hide the fact that predictions come from statistical models.

---

# Model Evaluation Metrics

Different metrics are used for classification and regression.

## Accuracy

Accuracy is:

```text
Correct Predictions
--------------------
Total Predictions
```

It measures the overall fraction of correct classifications.

---

## Precision

Precision answers:

> Of the songs predicted as hits, how many were actually hits?

```text
Precision =
True Positives
-------------------------
True Positives + False Positives
```

---

## Recall

Recall answers:

> Of all actual hits, how many did the model identify?

```text
Recall =
True Positives
-------------------------
True Positives + False Negatives
```

---

## F1 Score

F1 combines precision and recall.

```text
F1 =
2 × Precision × Recall
----------------------
Precision + Recall
```

It is useful when both false positives and false negatives matter.

---

## ROC-AUC

ROC-AUC stands for:

> Receiver Operating Characteristic - Area Under the Curve

It measures how well the classifier separates the positive and negative classes across different probability thresholds.

A higher value generally indicates stronger ranking/separation performance.

---

## MAE

MAE stands for:

> Mean Absolute Error

It measures the average absolute difference between predicted and actual numerical outcomes.

```text
MAE =
average(|actual - predicted|)
```

Lower is better.

---

## RMSE

RMSE stands for:

> Root Mean Squared Error

It is the square root of mean squared prediction error.

RMSE gives more weight to larger errors than MAE.

Lower is better.

---

## R²

R² stands for:

> Coefficient of Determination

It indicates how much of the variation in the target is explained by the regression model relative to a baseline.

Higher values generally indicate better explanatory predictive performance.

---

# Confusion Matrix

A confusion matrix is used for classification evaluation.

It contains four main outcomes:

```text
                    Actual
                 0          1
              ------------------
Predicted 0 |  TN       |   FN
Predicted 1 |  FP       |   TP
```

Where:

- TN = True Negative
- FN = False Negative
- FP = False Positive
- TP = True Positive

This allows the model's errors to be inspected instead of looking only at accuracy.

---

# Feature Importance

Random Forest models can provide feature-importance estimates.

The dashboard uses this to show which features were important to the trained model.

A conceptual example:

```text
Momentum       ███████████████
Engagement     ███████████
Daily Streams  █████████
Growth Rate    ███████
Exposure       █████
```

Feature importance is a global model-level diagnostic.

It should not be confused with the local feature-impact explanation for an individual song.

---

# Horizon Analysis

Horizon analysis examines model behavior across the future prediction horizon represented in the dataset.

This helps investigate whether the model's predictions remain meaningful as the forecasting window changes.

The exact horizon interpretation depends on the target and data generation process used in the project.

---

# Django Architecture

The web application is implemented with Django.

A simplified architecture is:

```text
Browser
  ↓
Django URL Routing
  ↓
Views
  ↓
Business Logic
  ↓
Models / ML Prediction Layer
  ↓
SQLite Database + Model Artifacts
  ↓
Templates
  ↓
Browser
```

Django handles:

- Routing.
- Request processing.
- Templates.
- Database interaction.
- Forms.
- Administration.
- Application structure.

---

# Database and ORM

The project uses Django's Object-Relational Mapping (ORM).

ORM means:

> Object-Relational Mapping

It allows Python classes to represent database tables.

Conceptually:

```text
Python Model
     ↕
Django ORM
     ↕
SQLite Table
```

This allows the application to store song predictions and retrieve them for dashboards and intelligence pages.

---

# Prediction Architecture

The website does not retrain the model whenever the database is seeded.

The shipped prediction architecture is:

```text
Trained model (.pkl)
        │
        ├──────────────→ /predict/
        │                    │
        │                    ▼
        │              New prediction
        │
        └──────────────→ Existing catalog
                             │
                             ▼
                         Dashboard
```

The `.pkl` files contain serialized trained machine learning models.

They are used for inference.

---

# Model Retraining

The project includes a dedicated:

```text
/train/
```

workflow for supervised retraining.

This is important because machine learning should not be described as "learning automatically" merely because new predictions are generated.

The model can only learn from labelled examples.

The retraining system accepts a CSV containing the required model features and either:

- `hit` with values `0` and `1`, or
- `future_streams`, from which the system derives the hit label.

---

# Training Dataset Requirements

A valid training dataset must contain all 11 model features.

It must also provide a known outcome.

For example:

```text
streams
daily_streams
growth_rate
momentum
exposure
engagement
skip_rate
save_rate
share_rate
playlist_rate
search_rate
hit
```

Or it can contain:

```text
future_streams
```

instead of `hit`.

The key requirement is that the training example has an observed outcome.

---

# Combine vs Replace Training

Two retraining strategies are available.

## Combine

Combine is the recommended strategy.

```text
Existing Training Data
        +
New Labelled Data
        ↓
Combined Dataset
        ↓
Retrained Model
```

This allows the model to retain previously learned patterns while incorporating new labelled examples.

---

## Replace

Replace trains only on the uploaded dataset.

```text
Uploaded Dataset
       ↓
New Training Run
       ↓
New Model
```

This can be useful when the new dataset is intended to completely redefine the training population.

However, it can also remove useful information contained in the previous dataset.

---

# Learning From New Data

Suppose a song is entered into `/predict/`.

At prediction time:

```text
Features known
Outcome unknown
```

Therefore:

```text
Prediction ≠ Training
```

Later, if the actual outcome becomes known:

```text
Features
+
Observed outcome
        ↓
Labelled training example
```

That example can then be included in a retraining dataset.

This is the correct supervised-learning feedback loop.

---

# Model Backup and Rescoring

When retraining succeeds, the previous model is backed up under:

```text
model_artifacts/backups/
```

This provides a basic rollback mechanism.

After retraining, the application also rescored the existing song catalog so that the dashboard reflects the new active model.

Conceptually:

```text
Old Model
   ↓
Backup

New Training Data
   ↓
Retraining
   ↓
New Model
   ↓
Rescore Existing Songs
   ↓
Updated Dashboard
```

---

# Avoiding Data Leakage

Data leakage occurs when information that would not realistically be available at prediction time is accidentally used to train a model.

For example, if a model is supposed to predict future performance using today's information, it should not receive future outcomes as an input feature.

The correct separation is:

```text
Past / Current Information
        ↓
Model Input
```

while:

```text
Future Outcome
        ↓
Training Target
```

This distinction is essential for trustworthy trajectory prediction.

---

# Web Application

The Django application provides multiple interfaces.

Major workflows include:

```text
Dashboard
Predict
Train / Update Model
Song Intelligence
Model Evidence
Django Admin
```

Each interface serves a different purpose.

---

# Dashboard

The dashboard is the primary portfolio-level analytics interface.

It provides:

- Portfolio overview.
- Hit prediction distribution.
- Future performance analysis.
- Momentum analysis.
- Exposure analysis.
- Opportunity queues.
- Decliner queues.
- Language intelligence.
- Genre intelligence.
- Action mix.

The goal is to move from raw rows of data to visual decision support.

---

# Live Demo Refresh

The dashboard includes a live-demo refresh behavior.

This is intentionally a simulation.

It demonstrates how streaming analytics could change as new observations arrive without claiming that the application is connected to a real-time JioSaavn stream.

This distinction is important:

```text
Live Demo Simulation
        ≠
Real-Time Streaming API
```

---

# Decision-Support Workflow

A typical analyst workflow is:

```text
1. Open dashboard
        ↓
2. Inspect portfolio
        ↓
3. Find high-potential songs
        ↓
4. Compare exposure and momentum
        ↓
5. Open individual song intelligence
        ↓
6. Inspect model explanation
        ↓
7. Examine counterfactual signals
        ↓
8. Decide which songs deserve attention
```

The machine learning model supports the decision.

It does not replace the decision-maker.

---

# Example Workflow

Suppose the dashboard identifies:

```text
Song X

Current Streams: Medium
Momentum: High
Exposure: Low
Engagement: High
Hit Probability: High
Future Forecast: Strong
```

The system can classify Song X as a potential hidden opportunity.

The analyst can then inspect:

```text
Why is the hit probability high?
Which features contribute most?
Is exposure unusually low?
What happens to the prediction when momentum changes?
```

This produces a much more useful analysis than simply saying:

```text
Song X is predicted to be a hit.
```

---

# Django Admin

Django Admin provides an administrative interface for inspecting application data.

It can be used to:

- Inspect stored predictions.
- Review database records.
- Manage application data.
- Support development and debugging.

---

# Project Structure

A typical project structure is:

```text
JioSaavn_Song_Trajectory_Django_ULTIMATE/
│
├── manage.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── prediction datasets
│   └── supporting synthetic data
│
├── model_artifacts/
│   ├── music_hit_prediction_model.pkl
│   ├── future_streams_regressor.pkl
│   └── backups/
│
├── predictor/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── admin.py
│   ├── management/
│   ├── templates/
│   └── ...
│
├── project/
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── ...
│
└── db.sqlite3
```

The exact structure may contain additional Django files depending on the current project version.

---

# Technology Stack

## Python

Python is used for:

- Machine learning.
- Data processing.
- Django backend development.
- Model inference.

---

## Django

Django is the web framework used to build the application.

It provides:

- URL routing.
- Views.
- Templates.
- ORM.
- Forms.
- Administration.

---

## Scikit-learn

Scikit-learn provides the machine learning algorithms and evaluation utilities.

It is used for:

- Random Forest classification.
- Random Forest regression.
- Train/test splitting.
- Evaluation.
- Feature analysis.

---

## Pandas

Pandas is used for tabular data processing.

It provides DataFrame-based workflows for loading, transforming, filtering, and preparing datasets.

---

## NumPy

NumPy provides numerical array operations used by the data and machine learning pipeline.

---

## SQLite

SQLite is used as the local application database.

It is convenient for demonstration and local development because it does not require a separate database server.

---

# Installation on macOS

Open Terminal and run:

```bash
cd ~/Desktop
unzip JioSaavn_Song_Trajectory_Django_ULTIMATE.zip
cd JioSaavn_Song_Trajectory_Django_ULTIMATE

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

# Database Setup

Run:

```bash
python manage.py makemigrations predictor
python manage.py migrate
python manage.py seed_demo
python manage.py check
```

The `seed_demo` command populates the database using the included demonstration data.

---

# Running the Application

Start Django:

```bash
python manage.py runserver
```

Then open:

```text
http://127.0.0.1:8000/
```

The development server should display the application dashboard.

---

# If the Database Was Created Incorrectly

Only use this procedure if the application reports an error such as:

```text
no such table: predictor_songprediction
```

Run:

```bash
rm -f db.sqlite3

python manage.py migrate
python manage.py seed_demo
python manage.py runserver
```

Do **not** delete:

```text
data/
model_artifacts/
```

Those directories contain project data and trained model artifacts required by the demonstration application.

---

# Environment Variables

If the project is extended for deployment, environment variables can be used for configuration such as:

```text
SECRET_KEY
DEBUG
ALLOWED_HOSTS
DATABASE_URL
```

Secrets should not be committed to GitHub.

A `.env` file should normally be excluded from version control.

---

# Git and GitHub

Initialize Git if necessary:

```bash
git init
```

Check the current remote:

```bash
git remote -v
```

Remove an existing remote:

```bash
git remote remove origin
```

Add the repository:

```bash
git remote add origin https://github.com/Ornblue/Song-Trajectory-Prediction-And-Classification.git
```

Verify:

```bash
git remote -v
```

Add and commit:

```bash
git add .
git commit -m "Initial project commit"
```

Push:

```bash
git branch -M main
git push -u origin main
```

If GitHub rejects the push because the remote already contains commits, first inspect the remote history before deciding whether to merge, rebase, or replace it.

---

# Git Ignore

A suitable `.gitignore` is:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
.venv/
venv/
env/
ENV/

# Django
db.sqlite3
db.sqlite3-journal
*.log
staticfiles/
media/
uploads/

# Environment variables
.env
.env.*
!.env.example

# Jupyter
.ipynb_checkpoints/

# macOS
.DS_Store

# Windows
Thumbs.db
desktop.ini

# IDE
.vscode/
.idea/

# Python packaging
*.egg-info/
dist/
build/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temporary files
*.tmp
*.temp
*.bak
*.swp

# Model backups
model_artifacts/backups/

# Temporary training data
active_training_data.csv
```

Do not blindly add:

```gitignore
data/
model_artifacts/
```

if the shipped demo application requires those files to run.

---

# Model Training Philosophy

The project follows supervised learning.

Supervised learning means the model learns a relationship between:

```text
Input Features
        +
Known Target
        ↓
Learned Model
```

For example:

```text
Current song behavior
        +
Known future outcome
        ↓
Training example
```

After learning, the model can receive a new feature vector where the future outcome is unknown.

---

# Global vs Local Explainability

The project contains two different types of explanation.

## Global Explanation

Global feature importance asks:

> Which features are generally important to the model?

This describes model behavior across the training population.

---

## Local Explanation

Local explanation asks:

> Why did the model behave this way for this particular song?

This is more useful when investigating an individual prediction.

The project provides local feature-impact and counterfactual diagnostics for this purpose.

---

# Interpretation Rules

Predictions should always be interpreted as estimates.

For example:

```text
Hit probability = 85%
```

does not mean:

```text
The song will definitely become a hit.
```

It means:

```text
Given the learned patterns and supplied features,
the model assigns a high probability to the hit class.
```

Similarly:

```text
Predicted future streams = 2,000,000
```

does not guarantee exactly 2,000,000 future streams.

Machine learning predictions depend on:

- Training data.
- Feature quality.
- Distribution shift.
- Model assumptions.
- Input quality.
- Real-world behavior.

---

# Limitations

The project has several important limitations.

## Synthetic Data

The included data is synthetic/demo data.

Therefore, model performance on this dataset should not be interpreted as evidence of performance on real JioSaavn production data.

---

## No Real-Time JioSaavn Feed

The live dashboard behavior is simulated.

It is not a production streaming pipeline.

---

## Prediction Is Not Causation

Feature impact does not prove causality.

If the model assigns a positive impact to momentum, this does not prove that artificially increasing momentum will cause the predicted improvement.

---

## Distribution Shift

A model trained on historical data may become less accurate if future music consumption behavior changes significantly.

---

## Retraining Requires Labels

The system cannot learn correctly from predictions alone.

New labelled examples are required for supervised retraining.

---

# Future Improvements

Possible extensions include:

## Real-Time Streaming Integration

Connect the application to a real streaming/event pipeline.

```text
Streaming Events
      ↓
Message Queue
      ↓
Feature Store
      ↓
Prediction Service
      ↓
Dashboard
```

---

## Automated Retraining

A scheduled pipeline could periodically retrain models after enough new labelled outcomes become available.

---

## Advanced Time-Series Models

Future versions could explore:

- XGBoost.
- LightGBM.
- Gradient Boosting.
- Temporal models.
- Recurrent Neural Networks.
- Transformer-based time-series architectures.

These should be compared against the existing Random Forest baseline rather than adopted automatically.

---

## Better Explainability

Future versions could add methods such as:

- SHAP.
- Partial Dependence.
- Accumulated Local Effects.

These could provide deeper global and local interpretation.

---

## Production Database

SQLite is convenient for local development.

A deployed system could use:

- PostgreSQL.
- Managed cloud databases.
- Separate analytical warehouses.

---

## Model Monitoring

A production version could monitor:

```text
Prediction quality
Data drift
Feature drift
Model drift
Latency
Error rates
```

---

# Glossary

## AI

Artificial Intelligence.

A broad field involving systems that perform tasks associated with intelligent behavior.

---

## ML

Machine Learning.

A branch of AI where algorithms learn patterns from data.

---

## Classification

A machine learning problem where the output is a category.

Example:

```text
Hit / Not Hit
```

---

## Regression

A machine learning problem where the output is a continuous numerical value.

Example:

```text
Future Streams
```

---

## Random Forest

An ensemble algorithm based on multiple decision trees.

---

## Feature

An input variable supplied to a machine learning model.

Example:

```text
Momentum
```

---

## Target

The outcome the model is trained to predict.

Examples:

```text
hit
future_streams
```

---

## Inference

Using an already-trained model to generate a prediction.

---

## Retraining

Training a model again using additional or updated labelled data.

---

## ORM

Object-Relational Mapping.

A programming technique that maps application objects to database tables.

---

## API

Application Programming Interface.

A structured way for software components to communicate.

---

## CSV

Comma-Separated Values.

A common text-based tabular data format.

---

## PKL

A common file extension for Python Pickle serialization.

The project uses `.pkl` files to store trained model objects.

---

## pp

Percentage points.

For example:

```text
60% → 70%
```

is a change of:

```text
10 percentage points
```

not a 10% relative increase.

---

## ROC-AUC

Receiver Operating Characteristic - Area Under the Curve.

A classification metric used to measure ranking/separation performance.

---

## MAE

Mean Absolute Error.

Average absolute prediction error.

---

## RMSE

Root Mean Squared Error.

A regression error metric that penalizes larger errors more strongly.

---

## R²

Coefficient of Determination.

A regression metric describing how much variation is explained relative to a baseline.

---

# Project Summary

JioSaavn Song Trajectory Intelligence is an end-to-end machine learning decision-support platform built around the idea that a song's future cannot be understood from current popularity alone.

The system combines:

```text
Song Data
   ↓
Feature Engineering
   ↓
Random Forest Classification
   +
Random Forest Regression
   ↓
Hit Probability
   +
Future Streams
   ↓
Explainability
   ↓
Opportunity / Decliner Detection
   ↓
Portfolio Analytics
   ↓
Django Web Application
```

The most important conceptual distinction in the project is:

```text
Current Performance
        ≠
Future Trajectory
```

A song with moderate current performance may have strong future potential if its behavioral signals indicate acceleration.

Likewise, a highly popular song may require attention if its momentum and future forecast weaken.

The project therefore moves beyond simple popularity prediction and attempts to provide a complete analytical workflow:

```text
Observe
  ↓
Predict
  ↓
Explain
  ↓
Prioritize
  ↓
Act
```

That is the central purpose of the JioSaavn Song Trajectory Intelligence platform.

---

# Author

Built as a machine learning and Django portfolio project focused on:

- Machine Learning
- Predictive Analytics
- Feature Engineering
- Explainable AI
- Business Intelligence
- Django Web Development
- Decision Support Systems
