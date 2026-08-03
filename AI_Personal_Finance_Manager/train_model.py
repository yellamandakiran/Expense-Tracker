from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "dataset" / "finance_data.csv"
MODEL_FILE = BASE_DIR / "expense_model.pkl"
FEATURES = ["Income", "Food", "Transport", "Shopping", "Bills"]
TARGET = "TotalExpense"


def train_model():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    data = pd.read_csv(DATASET_FILE)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing dataset columns: {', '.join(missing_columns)}")

    data = data[required_columns].apply(pd.to_numeric, errors="coerce").dropna()
    data = data[(data[required_columns] >= 0).all(axis=1)]
    if len(data) < 10:
        raise ValueError("The dataset must contain at least 10 valid rows.")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    joblib.dump(model, MODEL_FILE)

    print("=" * 46)
    print("AI Personal Finance Manager - Model Training")
    print("=" * 46)
    print(f"Dataset rows used : {len(data)}")
    print(f"MAE               : {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R² score          : {r2_score(y_test, predictions):.4f}")
    print(f"Model saved to    : {MODEL_FILE.name}")
    print("=" * 46)


if __name__ == "__main__":
    train_model()
