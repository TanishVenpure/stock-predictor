import shap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import load_data
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
    roc_curve
)

os.makedirs("../reports", exist_ok=True)
matplotlib.rcParams["figure.dpi"] = 150


(X_train, X_test,
 y_train, y_test,
 FEATURE_COLS) = load_data()

final_stack = joblib.load("../models/stacking_ensemble.pkl")
best_rf     = joblib.load("../models/random_forest.pkl")
best_xgb    = joblib.load("../models/xgboost.pkl")
best_svm    = joblib.load("../models/svm.pkl")


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

models = [
    ("Random Forest",     best_rf),
    ("XGBoost",           best_xgb),
    ("SVM",               best_svm),
    ("Stacking Ensemble", final_stack)
]

for ax, (name, model) in zip(axes, models):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["DOWN", "UP"],
        colorbar=False,
        ax=ax,
        cmap="Blues"
    )
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    ax.set_title(f"{name}\nROC-AUC: {auc:.4f}", fontsize=11)

plt.suptitle("Confusion Matrices — All Models", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("../reports/confusion_matrix.png", bbox_inches="tight")
plt.show()
print("Saved confusion_matrix.png")



importance_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": best_xgb.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print(importance_df.head(20))

importance_df.to_csv(
    "../reports/feature_importance.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 6))

colors = ["#2196F3", "#FF9800", "#9C27B0", "#F44336"]

for (name, model), color in zip(models, colors):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})",
            color=color, linewidth=2)

ax.plot([0,1], [0,1], "k--", linewidth=1, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — All Models", fontsize=13)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../reports/roc_curve.png", bbox_inches="tight")
plt.show()
print("Saved roc_curve.png")

print("Computing SHAP values...")

# use 500 test samples for speed
X_shap = X_test[:500]

explainer   = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_shap)

# shap_values is [class_0, class_1] — use class_1 (UP)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

print(f"SHAP values shape: {sv.shape}")

plt.figure(figsize=(10, 8))
shap.summary_plot(
    sv,
    X_shap,
    feature_names=FEATURE_COLS,
    plot_type="dot",
    max_display=20,
    show=False
)
plt.title("SHAP Feature Importance — Random Forest", fontsize=13)
plt.tight_layout()
plt.savefig("../reports/shap_summary.png", bbox_inches="tight")
plt.show()
print("Saved shap_summary.png")



plt.figure(figsize=(10, 7))
shap.summary_plot(
    sv,
    X_shap,
    feature_names=FEATURE_COLS,
    plot_type="bar",
    max_display=15,
    show=False
)
plt.title("Top 15 Features by Mean SHAP Value", fontsize=13)
plt.tight_layout()
plt.savefig("../reports/feature_importance.png", bbox_inches="tight")
plt.show()
print("Saved feature_importance.png")

explainer_xgb   = shap.TreeExplainer(best_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_shap)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values_xgb,
    X_shap,
    feature_names=FEATURE_COLS,
    plot_type="bar",
    max_display=15,
    show=False
)
plt.title("Top 15 Features by Mean SHAP Value — XGBoost", fontsize=13)
plt.tight_layout()
plt.savefig("../reports/shap_xgb.png", bbox_inches="tight")
plt.show()

sample_idx = 0
shap_exp = explainer(pd.DataFrame(X_shap[:10],
                                   columns=FEATURE_COLS))

plt.figure()
sample_idx = 0

shap_exp = explainer(
    pd.DataFrame(
        X_shap,
        columns=FEATURE_COLS
    )
)

# Use class 1 (UP)
shap.plots.waterfall(
    shap_exp[sample_idx, :, 1],
    show=False
)

plt.tight_layout()

plt.savefig(
    "../reports/shap_waterfall.png",
    bbox_inches="tight"
)

plt.show()


print("Saved shap_waterfall.png")


rows = []
for name, model in models:
    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    rows.append({
        "model"    : name,
        "accuracy" : round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall"   : round(recall_score(y_test, pred), 4),
        "f1"       : round(f1_score(y_test, pred), 4),
        "roc_auc"  : round(roc_auc_score(y_test, proba), 4)
    })

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("../reports/model_metrics.csv", index=False)
print(metrics_df.to_string(index=False))

#error annalysis

preds = final_stack.predict(X_test)

probs = final_stack.predict_proba(X_test)[:,1]
errors_df = X_test.copy()

errors_df["actual"] = y_test.values

errors_df["predicted"] = preds

errors_df["correct"] = (
    errors_df["actual"]
    ==
    errors_df["predicted"]
)
mistakes = errors_df[
    errors_df["correct"] == False
]

mistakes.to_csv(
    "../reports/error_analysis.csv",
    index=False
)

print(
    f"Total mistakes: {len(mistakes)}"
)

ticker_cols = [
    c for c in FEATURE_COLS
    if c.startswith("ticker_")
]

errors_df["ticker"] = (
    errors_df[ticker_cols]
    .idxmax(axis=1)
)

stock_error = (
    errors_df
    .groupby("ticker")["correct"]
    .mean()
    .sort_values()
)

print("\nAccuracy by stock")

print(stock_error)

stock_error.to_csv(
    "../reports/stock_accuracy.csv"
)

print("\nAll reports saved to reports/" )