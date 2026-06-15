import pandas as pd
import numpy as np
import joblib
import os
from data_loader import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

tscv = TimeSeriesSplit(
    n_splits=5
)

(
    X_train,
    X_test,
    y_train,
    y_test,
    FEATURE_COLS
) = load_data()

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,           
    min_samples_split=100, 
    min_samples_leaf=50,   
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=50,       # ← reduce from 100 to 50
    max_depth=2,           # ← reduce from 3 to 2
    learning_rate=0.01,
    subsample=0.5,         # ← reduce from 0.6
    colsample_bytree=0.5,  # ← reduce from 0.6
    min_child_weight=20,   # ← increase from 10
    gamma=2.0,             # ← increase from 1.0
    reg_alpha=1.0,         # ← increase from 0.5
    reg_lambda=5.0,        # ← increase from 2.0
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
svm = SVC(
    C=9.897745563620338,
    gamma=0.016403654058008886,
    kernel="rbf",
    probability=True,
    random_state=42
)

meta_learner = LogisticRegression(
    C=0.01,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

stacking_clf = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb),
        ("svm", svm)
    ],
    final_estimator=meta_learner,
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1
)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
svm.fit(X_train, y_train)
stacking_clf.fit(X_train, y_train)


os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/random_forest.pkl")

joblib.dump(xgb, "models/xgboost.pkl")

joblib.dump(svm, "models/svm.pkl")

joblib.dump(
    stacking_clf,
    "models/stacking_ensemble.pkl"
)

joblib.dump(
    FEATURE_COLS,
    "../models/feature_cols.pkl"
)

def evaluate_model(name, model):

    pred = model.predict(X_test)

    proba = model.predict_proba(X_test)[:,1]

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, proba), 4)
    }

results = []

results.append(evaluate_model("Random Forest", rf))
results.append(evaluate_model("XGBoost", xgb))
results.append(evaluate_model("SVM", svm))
results.append(evaluate_model("Stacking Ensemble", stacking_clf))

metrics_df = pd.DataFrame(results)

metrics_df = metrics_df.sort_values(
    by="ROC_AUC",
    ascending=False
)

print(metrics_df)

preds = xgb.predict(X_test)

print(pd.Series(preds).value_counts(normalize=True))

print(y_test.value_counts(normalize=True))

probs = xgb.predict_proba(X_test)[:,1]

print(probs.min())
print(probs.max())
print(probs.mean())

from sklearn.metrics import roc_auc_score

probs = xgb.predict_proba(X_test)[:,1]

for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:

    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(
        f"Threshold={threshold:.2f} | "
        f"Acc={acc:.4f} | "
        f"Prec={prec:.4f} | "
        f"Recall={rec:.4f} | "
        f"F1={f1:.4f}"
    )



for name, model in [("RF", rf), ("XGB", xgb), ("SVM", svm)]:
    model.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
    test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:,1])
    print(f"{name:<5} Train: {train_auc:.4f}  Test: {test_auc:.4f}  "
          f"Gap: {train_auc - test_auc:.4f}")

