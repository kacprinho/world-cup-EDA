# world-cup-EDA

## Overview

This project explores FIFA World Cup 2026 player and match statistics, with a focus on EDA and comparing several machine learning models for predicting the number of goals scored. The goal of the project isn't to build the most accurate predictor possible, but to test and compare different modeling approaches against a common set of features and evaluation metrics, as well as get familiar with the workflow that comes with this sort of project.

## Problem
Given multiple other statistics, predict the number of goals a player scores.

## Data Sources
Kaggle dataset — https://www.kaggle.com/datasets/swaptr/fifa-wc-2026-players

## Project Structure
```
├── data/
│   └── players.csv              # original, unmodified source data
├── notebooks/
│   ├── player_EDA.ipynb         # exploratory analysis, missingness, visualizations
│   └── player-ML.ipynb          # model training, evaluation, tuning
├── src/
│   └── model_evaluation.py      # reusable functions for model evaluation 
├── .gitignore
├── requirements.txt
└── README.md
```

## Results
Final results can be found in the last bar chart at the end of the second player-ML notebook.
We see that Ridge Regression comes out on top with the lowest MSE, which is to be expected.

## Notes
random_state=13 is used consistently across all models and splits for reproducibility.

## Tech stack
Python 3.13 · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## Git Workflow
- main contains stable, working notebooks/code only
- Feature branches (data-collection, ML-model, etc.) for in-progress work, merged into main via pull request once complete
- Commit messages follow a short imperative style

## Author
Kacper Kawecki — Discrete Mathematics student, University of Warwick