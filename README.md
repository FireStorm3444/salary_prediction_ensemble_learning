# 💰 AI Job Salary Prediction App

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-green)](https://salary-prediction-ensemble-learning.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Streamlit-powered web application that predicts the expected annual salary for AI and data-related roles using an ensemble learning model trained on real-world job market data.

🌐 **Live Demo**: [Click here to try it out!](https://salary-prediction-ensemble-learning.streamlit.app/)

---

## 🚀 Features

- Predicts annual, monthly, and hourly salary
- Interactive form to input job and company details
- Built with `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBoost`, `StackingRegressor` and pre-trained on an extensive AI job salary dataset
- Supports multiple countries, job roles, industries, and education levels
- Cached model loading for faster inference

---

## 🧠 How It Works

The app takes 10+ job-related inputs such as:

- Job title
- Experience level
- Company location & size
- Remote work ratio
- Education required
- Years of experience
- Industry

...and passes them through a preprocessing pipeline (including ordinal encoding and scaling) before making a salary prediction using a log-transformed regression model.

---

## 🗂️ Project Structure

```

├── app.py                  # Main Streamlit application
├── models/
│   ├── salary\_prediction\_model.pkl  # Trained ML model (via Git LFS)
│   ├── scaler.pkl                  # StandardScaler instance
│   └── encoder.pkl                 # OrdinalEncoder instance
├── requirements.txt
└── README.md

````

---

## 📦 Setup Instructions

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
````

### ▶️ Run Locally

```bash
streamlit run app.py
```

### 📦 Or Clone & Run with Git LFS Support

```bash
git lfs install
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
streamlit run app.py
```

> ⚠️ Make sure `Git LFS` is installed to download the large model files.

---

## 📊 Model Details

* **Algorithms Used**: 
  * Random Forest Regressor
  * Gradient Boosting Regressor
  * XGBoost Regressor
  * Stacking Regressor (Ensemble Learning)
* **Preprocessing**:

  * Ordinal encoding for categorical features
  * Standard scaling for numerical features
  * Log transformation of the salary target
* **Trained on**: 2020–2025 AI/ML job postings dataset

---

## 📈 Example Prediction Output

| Metric           | Example Value    |
| ---------------- | ---------------- |
| Predicted Salary | \$123,456/year   |
| Monthly Salary   | \~\$10,288/month |
| Hourly Rate      | \~\$59/hour      |

---

## 📃 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

Developed by **Shekhar**
🔗 [LinkedIn](https://www.linkedin.com/in/shekhar-coder/)
📫 Contact: [shekhar99bd@gmail.com](mailto:shekhar.ai.projects@gmail.com) *(example)*

---

> 🌟 Star this repo if you like the project — and feel free to contribute or fork!