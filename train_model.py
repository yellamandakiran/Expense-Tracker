import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Create Sample Dataset
# -----------------------------

np.random.seed(42)

data = pd.DataFrame({
    "Income": np.random.randint(25000, 150000, 500),
    "Food": np.random.randint(2000, 15000, 500),
    "Transport": np.random.randint(1000, 10000, 500),
    "Shopping": np.random.randint(1000, 20000, 500),
    "Bills": np.random.randint(2000, 12000, 500)
})

# Target: Total Monthly Expense
data["TotalExpense"] = (
    data["Food"] +
    data["Transport"] +
    data["Shopping"] +
    data["Bills"] +
    np.random.randint(-1000, 1000, 500)
)

# -----------------------------
# Features and Target
# -----------------------------

X = data[[
    "Income",
    "Food",
    "Transport",
    "Shopping",
    "Bills"
]]

y = data["TotalExpense"]

# -----------------------------
# Train Model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "expense_model.pkl")

print("======================================")
print(" AI Personal Finance Manager")
print("======================================")
print("Model trained successfully.")
print("Saved as: expense_model.pkl")
print("Dataset Size:", len(data))
print("======================================")