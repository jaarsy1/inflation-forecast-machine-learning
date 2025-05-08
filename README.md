# 📈 French Inflation Forecasting with Machine Learning

This project aims to predict French inflation using macroeconomic and financial variables. We compare traditional econometric approaches (ARIMA, OLS) with machine learning models (XGBoost, Random Forest) using data from Bloomberg (until Feb 2025).

---

## 📌 Project Objectives

- Analyze inflation trends and components
- Apply and compare several predictive models:
  - ARIMA (1,1,1)
  - OLS Regression with lags
  - XGBoost Regressor
  - Random Forest Regressor
- Identify model performance and limitations
- Provide insights for economic decision-making

---

## 📂 Repository Structure

```
inflation-forecast-ml/
├── data/                  # Raw macroeconomic data (Bloomberg)
├── src/                   # All modeling scripts
│   └── inflation_forecasting.py
├── report/                # Final academic report
│   └── ML_Report.pdf
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🔧 How to Run

1. Clone the repository:

```bash
git clone https://github.com/jaarsy1/inflation-forecast-ml.git
cd inflation-forecast-ml
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main script:

```bash
python src/inflation_forecasting.py
```

> ⚠️ Note: The Bloomberg data is provided for academic purposes and should not be redistributed.

---

## 📊 Model Performance Summary

| Model       | RMSE   | MAE    | R²     |
|-------------|--------|--------|--------|
| ARIMA(1,1,1)| 0.9291 | 0.8296 | -1.5758|
| OLS         | 0.1829 | 0.2404 | 0.4928 |
| XGBoost     | 2.0708 | 1.4451 | 0.2034 |

*See the report for more details.*

---

## 🧠 Authors

- Gabriel B.  
- Simon D.  
- Enzo N.

> MSc Finance – ESSEC Business School 
> Course: **Supervised Learning for Finance** with Prof. Elise G. (2025)

---

## 📃 License

This project is for **academic use only**.
