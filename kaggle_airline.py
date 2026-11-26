import optuna
import pandas as pd
from catboost import CatBoostClassifier
from optuna.integration import CatBoostPruningCallback
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True)


train = pd.read_csv("data/train.csv")

x = train[["Age","Flight Distance","Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment","On-board service","Leg room service","Baggage handling","Checkin service","Cleanliness","Departure Delay in Minutes","Arrival Delay in Minutes","Gender","Customer Type","Type of Travel","Class"]].copy()
y = train["satisfaction"]
text_cols = x.select_dtypes(exclude=["number", "bool"]).columns

for col in text_cols:
    cats = sorted(x[col].dropna().unique())
    x[col] = pd.Categorical(x[col], categories=cats)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(x_train, y_train)

best_params = {
    "learning_rate": 0.03963012169509674,
    "depth": 8,
    "l2_leaf_reg": 1.4338195112127787,
    "random_strength": 1.7812773834198012,
    "bagging_temperature": 0.9082104616217752,
}

def objective(trial):
    params = {
        "iterations": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
        "random_strength": trial.suggest_float("random_strength", 0, 5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "border_count": 128, 
        "eval_metric": "Logloss",
        "early_stopping_rounds": 100,
        "allow_writing_files": False,
        "verbose": 0,
    }
    pruner = CatBoostPruningCallback(trial, "Logloss")
    cb = CatBoostClassifier(**params)
    cb.fit(x_train, y_train, cat_features=list(text_cols), eval_set=(x_val, y_val), callbacks=[pruner])
    pruner.check_pruned()
    return roc_auc_score(y_val, cb.predict_proba(x_val)[:, 1])


TUNE = False

if TUNE:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, timeout=1200)
    print(study.best_params)

#best_params = hardcoded params

best_cb = CatBoostClassifier(iterations=3000, border_count=128, eval_metric="Logloss",
                             early_stopping_rounds=100, allow_writing_files=False,
                             verbose=0, **best_params)
best_cb.fit(x_train, y_train, cat_features=list(text_cols), eval_set=(x_val, y_val))

hgb_p = model.predict_proba(x_val)[:, 1]
cb_p  = best_cb.predict_proba(x_val)[:, 1]
ens_p = (hgb_p + cb_p) / 2

print("HGB", roc_auc_score(y_val, hgb_p))
print("CB ", roc_auc_score(y_val, cb_p))
print("ENS", roc_auc_score(y_val, ens_p))