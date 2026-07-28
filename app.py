from pathlib import Path
import sqlite3

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

app = Flask(__name__)
app.secret_key = "personal-finance-manager"

BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "finance.db"
MODEL_FILE = BASE_DIR / "expense_model.pkl"


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
                amount REAL NOT NULL,
                transaction_date TEXT NOT NULL,
                description TEXT
            )
            """
        )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/add", methods=["POST"])
def add_transaction():
    transaction_type = request.form.get("transaction_type")
    category = request.form.get("category")
    amount = request.form.get("amount")
    transaction_date = request.form.get("transaction_date")
    description = request.form.get("description", "").strip()

    if not transaction_type or not category or not amount or not transaction_date:
        flash("Please fill in all required fields.")
        return redirect(url_for("home"))

    try:
        amount = float(amount)

        if amount <= 0:
            raise ValueError

    except ValueError:
        flash("Enter a valid amount greater than zero.")
        return redirect(url_for("home"))

    with get_database() as connection:
        connection.execute(
            """
            INSERT INTO transactions (
                transaction_type,
                category,
                amount,
                transaction_date,
                description
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transaction_type,
                category,
                amount,
                transaction_date,
                description,
            ),
        )

    flash("Transaction added successfully.")
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    with get_database() as connection:
        transactions = connection.execute(
            """
            SELECT *
            FROM transactions
            ORDER BY transaction_date DESC, id DESC
            """
        ).fetchall()

        total_income = connection.execute(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM transactions
            WHERE transaction_type = 'Income'
            """
        ).fetchone()[0]

        total_expense = connection.execute(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM transactions
            WHERE transaction_type = 'Expense'
            """
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

    balance = total_income - total_expense

    category_names = [row["category"] for row in category_expenses]
    category_values = [round(row["total"], 2) for row in category_expenses]

    return render_template(
        "dashboard.html",
        transactions=transactions,
        total_income=round(total_income, 2),
        total_expense=round(total_expense, 2),
        balance=round(balance, 2),
        category_names=category_names,
        category_values=category_values,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    predicted_expense = None
    savings = None
    message = None

    if request.method == "POST":
        try:
            income = float(request.form["income"])
            food = float(request.form["food"])
            transport = float(request.form["transport"])
            shopping = float(request.form["shopping"])
            bills = float(request.form["bills"])

            if min(income, food, transport, shopping, bills) < 0:
                raise ValueError

            if not MODEL_FILE.exists():
                message = "Train the model first by running train_model.py."
            else:
                model = joblib.load(MODEL_FILE)

                input_data = pd.DataFrame(
                    [
                        {
                            "Income": income,
                            "Food": food,
                            "Transport": transport,
                            "Shopping": shopping,
                            "Bills": bills,
                        }
                    ]
                )

                predicted_expense = round(
                    float(model.predict(input_data)[0]),
                    2,
                )

                savings = round(income - predicted_expense, 2)

                if savings > 0:
                    message = "Your estimated monthly savings are positive."
                elif savings == 0:
                    message = "Your estimated income and expenses are equal."
                else:
                    message = "Your predicted expenses are higher than your income."

        except ValueError:
            flash("Enter valid positive numbers.")
            return redirect(url_for("predict"))

        except Exception as error:
            message = f"Prediction failed: {error}"

    return render_template(
        "prediction.html",
        predicted_expense=predicted_expense,
        savings=savings,
        message=message,
    )


@app.route("/delete/<int:transaction_id>", methods=["POST"])
def delete_transaction(transaction_id):
    with get_database() as connection:
        connection.execute(
            "DELETE FROM transactions WHERE id = ?",
            (transaction_id,),
        )

    flash("Transaction deleted successfully.")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    create_database()
    app.run(debug=True)