# 💰 AI Personal Finance Manager

A modern **Flask + Machine Learning** web application that helps users manage their finances by recording income and expenses while predicting future monthly expenses using Artificial Intelligence.

---

# 📌 Features

* Add Income Transactions
* Add Expense Transactions
* View Financial Dashboard
* Track Income & Expenses
* View Current Balance
* Delete Transactions
* AI-Based Expense Prediction
* Simple and Responsive User Interface
* SQLite Database
* Machine Learning using Random Forest

---

# 🛠 Technologies Used

* Python
* Flask
* HTML5
* CSS3
* SQLite
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

# 📂 Project Structure

```
AI_Personal_Finance_Manager/

│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── model/
│   └── expense_model.pkl
│
├── database/
│   └── finance.db
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── prediction.html
│
└── dataset/
    └── finance_data.csv
```

---

# 📥 Installation

### Step 1

Clone the repository

```
git clone <repository-link>
```

---

### Step 2

Move into the project folder

```
cd AI_Personal_Finance_Manager
```

---

### Step 3

Create Virtual Environment

```
python -m venv venv
```

---

### Step 4

Activate Virtual Environment

**Windows**

```
venv\Scripts\activate
```

**Linux / macOS**

```
source venv/bin/activate
```

---

### Step 5

Install Dependencies

```
pip install -r requirements.txt
```

---

### Step 6

Train the Machine Learning Model

```
python train_model.py
```

This creates:

```
expense_model.pkl
```

---

### Step 7

Run the Flask Application

```
python app.py
```

---

### Step 8

Open your browser

```
http://127.0.0.1:5000
```

---

# 📊 Dashboard

The dashboard displays:

* Total Income
* Total Expense
* Current Balance
* Transaction History

---

# 🤖 AI Prediction

The prediction module estimates monthly expenses using:

* Monthly Income
* Food Expense
* Transport Expense
* Shopping Expense
* Bills

---

# 📦 Required Packages

```
Flask
pandas
numpy
scikit-learn
joblib
matplotlib
```

---


# 🚀 Future Enhancements

* User Login System
* Multiple User Accounts
* Monthly Reports
* Data Visualization Charts
* Export to PDF
* Email Reports
* Dark Mode
* Cloud Deployment

---
