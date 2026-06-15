import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

DROP_COLS = [
    "Date", "Open", "High", "Low", "Close", "Volume",
    "tomorrow_close", "ticker", "target"
]

def load_data(csv_path=None, train_cutoff="2023-06-08"):

    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "all_stocks_features.csv"
        )

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # ── sanity check date range ──────────────────────────
    print(f"Data range : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Total rows : {len(df)}")

    # ── one-hot encode ticker ────────────────────────────
    df = pd.get_dummies(df, columns=["ticker"], dtype=int)

    # ── feature columns (everything not in DROP_COLS) ────
    FEATURE_COLS = [c for c in df.columns
                    if c not in DROP_COLS + ["target"]]

    print(f"Total Features : {len(FEATURE_COLS)}")
    print(f"Features       : {FEATURE_COLS}")

    # ── time-based split ─────────────────────────────────
    train = df[df["Date"] <= train_cutoff]
    test  = df[df["Date"] >  train_cutoff]

    print(f"\nTrain : {len(train)} rows | "
          f"{train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"Test  : {len(test)} rows  | "
          f"{test['Date'].min().date()} → {test['Date'].max().date()}")

    # ── check train is larger than test ──────────────────
    assert len(train) > len(test), (
        f"Train ({len(train)}) smaller than test ({len(test)}) — "
        f"adjust train_cutoff"
    )

    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_test,  y_test  = test[FEATURE_COLS],  test["target"]

    print(f"\nTrain UP% : {y_train.mean()*100:.1f}%")
    print(f"Test  UP% : {y_test.mean()*100:.1f}%")

    # ── scale ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── save scaler and feature cols ──────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler,       "models/scaler.pkl")
    joblib.dump(FEATURE_COLS, "models/feature_cols.pkl")
    print("\nScaler and feature_cols saved")

    return X_train_scaled, X_test_scaled, y_train, y_test, FEATURE_COLS