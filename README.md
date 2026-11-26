# airline-satisfaction
Predicts whether an airline passenger is satisfied using HistGradientBoosting and CatBoost, tuned with Optuna.

# Results
| Model    | AUC    |
|----------|--------|
| HGB      | 0.9572 |
| CatBoost | 0.9577 |
| Ensemble | 0.9577 |

Optuna tuning (40 trials) gave no real gain over the defaults suggesting the data has hit its ceiling.
