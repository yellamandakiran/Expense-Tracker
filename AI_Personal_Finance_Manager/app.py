import os
import sqlite3
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "finance-manager-development-key")

BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "finance.db"
MODEL_FILE = BASE_DIR / "expense_model.pkl"

ALLOWED_TYPES = {"Income", "Expense"}
ALLOWED_CATEGORIES = {
    "Salary", "Food", "Transport", "Shopping", "Bills",
    "Entertainment", "Medical", "Education", "Others",
}


def get_database():
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def create_database():
    with get_database() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_type TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                transaction_date TEXT NOT NULL,
                description TEXT
            )
            """
        )


create_database()


@app.template_filter("currency")
def currency(value):
    return f"₹{float(value):,.2f}"


@app.route("/")
def home():
    return render_template("index.html", today=date.today().isoformat())


@app.route("/add", methods=["POST"])
def add_transaction():
    transaction_type = request.form.get("transaction_type", "").strip()
    category = request.form.get("category", "").strip()
    amount_text = request.form.get("amount", "").strip()
    transaction_date = request.form.get("transaction_date", "").strip()
    description = request.form.get("description", "").strip()[:250]

    if transaction_type not in ALLOWED_TYPES:
        flash("Select a valid transaction type.")
        return redirect(url_for("home") + "#transaction-form")

    if category not in ALLOWED_CATEGORIES:
        flash("Select a valid category.")
        return redirect(url_for("home") + "#transaction-form")

    try:
        amount = float(amount_text)
        if amount <= 0:
            raise ValueError
    except ValueError:
        flash("Enter a valid amount greater than zero.")
        return redirect(url_for("home") + "#transaction-form")

    try:
        date.fromisoformat(transaction_date)
    except ValueError:
        flash("Select a valid transaction date.")
        return redirect(url_for("home") + "#transaction-form")

    with get_database() as connection:
        connection.execute(
            """
            INSERT INTO transactions
                (transaction_type, category, amount, transaction_date, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (transaction_type, category, amount, transaction_date, description),
        )

    flash("Transaction added successfully.")
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    with get_database() as connection:
        transactions = connection.execute(
            "SELECT * FROM transactions ORDER BY transaction_date DESC, id DESC"
        ).fetchall()

        total_income = connection.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE transaction_type = 'Income'"
        ).fetchone()[0]

        total_expense = connection.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE transaction_type = 'Expense'"
        ).fetchone()[0]

        category_expenses = connection.execute(
            """
            SELECT category, SUM(amount) AS total
            FROM transactions
            WHERE transaction_type = 'Expense'
            GROUP BY category
            ORDER BY total DESC
            """
        ).fetchall()

    return render_template(
        "dashboard.html",
        transactions=transactions,
        total_income=total_income,
        total_expense=total_expense,
        balance=total_income - total_expense,
        category_names=[row["category"] for row in category_expenses],
        category_values=[round(row["total"], 2) for row in category_expenses],
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form_values = {key: "" for key in ["income", "food", "transport", "shopping", "bills"]}

    if request.method == "POST":
        try:
            values = {}
            for key in form_values:
                raw_value = request.form.get(key, "").strip()
                form_values[key] = raw_value
                values[key] = float(raw_value)

            if any(value < 0 for value in values.values()):
                raise ValueError

            if values["income"] <= 0:
                flash("Monthly income must be greater than zero.")
                return render_template("prediction.html", result=None, form_values=form_values)

            if not MODEL_FILE.exists():
                result = {"error": "Model file is missing. Run: python train_model.py"}
            else:
                model = joblib.load(MODEL_FILE)
                input_data = pd.DataFrame([{
                    "Income": values["income"],
                    "Food": values["food"],
                    "Transport": values["transport"],
                    "Shopping": values["shopping"],
                    "Bills": values["bills"],
                }])

                predicted_expense = max(0, float(model.predict(input_data)[0]))
                savings = values["income"] - predicted_expense

                if savings > 0:
                    message = "Your estimated monthly savings are positive."
                elif savings == 0:
                    message = "Your estimated income and expenses are equal."
                else:
                    message = "Your predicted expenses are higher than your income."

                result = {
                    "predicted_expense": predicted_expense,
                    "savings": savings,
                    "message": message,
                }
        except (TypeError, ValueError):
            flash("Enter valid non-negative numbers in every field.")
        except Exception as error:
            result = {"error": f"Prediction failed: {error}"}

    return render_template("prediction.html", result=result, form_values=form_values)


@app.route("/delete/<int:transaction_id>", methods=["POST"])
def delete_transaction(transaction_id):
    with get_database() as connection:
        cursor = connection.execute(
            "DELETE FROM transactions WHERE id = ?", (transaction_id,)
        )

    flash("Transaction deleted successfully." if cursor.rowcount else "Transaction not found.")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
