# 📦 M5 Walmart Demand Forecasting

A machine learning-based demand forecasting application that predicts future daily sales for Walmart products across multiple stores.

The project uses **HistGradientBoostingRegressor** with lag features, rolling statistics, and calendar-based features to generate **28-day recursive sales forecasts**. An interactive **Streamlit dashboard** allows users to explore historical demand, forecasted sales, model performance, and feature importance.

---

## 🚀 Live Demo

🔗 **[Open the Streamlit App](https://m5demandforecastinggit-bc6ydjkukyfrlrpmnlzeqs.streamlit.app/)**

🔗 **[GitHub Repository](https://github.com/CodeMaestro-sys/M5_Demand_Forecasting.git)**

---

## ✨ Features

- Explore historical daily sales across selected products and stores
- Generate and visualize 28-day demand forecasts
- Compare historical sales with forecasted demand
- Evaluate model performance using MAE and RMSE
- Compare machine learning predictions against a seasonal naive baseline
- Explore feature importance and model insights
- Interactive Streamlit dashboard with multiple sections

---

## 🧠 Machine Learning Approach

The project uses historical sales data and time-based features to predict future product demand.

```text
Historical Sales Data
        ↓
Data Preparation & Filtering
        ↓
Feature Engineering
        ↓
Lag Features + Rolling Statistics
        ↓
Calendar, Event & SNAP Features
        ↓
HistGradientBoostingRegressor
        ↓
Recursive 28-Day Forecast
        ↓
Forecast Visualization
```

### Feature Engineering

The model uses the following features:

- **Lag features:** Sales from 1, 7, 14, and 28 days earlier
- **Rolling statistics:** Rolling mean over 7, 14, and 28 days
- **Calendar features:** Date-based information such as day and month
- **Event features:** Calendar event indicators
- **SNAP features:** SNAP program indicators
- **Categorical features:** Product and store identifiers

### Model Evaluation

The model was evaluated against a seasonal naive forecasting baseline.

| Model | MAE | RMSE |
|---|---:|---:|
| Seasonal Naive | 1.0527 | 2.5547 |
| HistGradientBoostingRegressor | 0.8758 | 1.7917 |

On the reported validation window, the machine learning model achieved approximately **16.8% lower MAE** and **29.9% lower RMSE** than the seasonal naive baseline.

Additional multi-window validation produced:

- **Mean MAE:** 0.8557
- **MAE standard deviation:** 0.0195
- **Mean RMSE:** 1.8612
- **RMSE standard deviation:** 0.0952

*Note: These results apply to the evaluated validation windows and selected subset of product-store series; they should not be interpreted as performance across the entire M5 competition dataset.*

---

## 📊 Dataset

The project uses the **M5 Forecasting dataset**, which contains Walmart historical sales, calendar events, and product pricing information.

The forecasting workflow uses historical sales from `sales_train_validation.csv`, covering the first 1,913 daily observations per product-store series. This avoids using the competition's evaluation-period sales as historical training data.

### Project Data Subset

To keep the project manageable, the modeling workflow uses:

- **10 Walmart stores**
- **100 unique products**
- **1,000 product-store series**
- **1,913 historical daily observations per series**
- **1,913,000 rows** after converting the selected sales data into long format

The generated forecast covers **28 days: April 25 to May 22, 2016**.

Dataset source: [M5 Forecasting – Accuracy on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

The original competition files are not included in this repository.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Streamlit
- Jupyter Notebook
- Git & GitHub

---

## 📁 Project Structure

```text
Demand-Forecasting/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── M5_Demand_Forecasting_Final.ipynb
│
├── m5_demand_forecasting_model.joblib
├── m5_28_day_forecast.csv
├── m5_historical_sample_small.csv
                     # Original dataset files, excluded from GitHub
```


---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/CodeMaestro-sys/M5_Demand_Forecasting.git
```

### 2. Navigate to the project folder

```bash
cd Demand-Forecasting
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application loads the saved model, forecast, and historical sample files. It does not retrain the model when the dashboard starts.

---

## 💼 Business Use Case

Demand forecasting helps retailers make better inventory and replenishment decisions by estimating how much of each product customers may purchase in the coming days.

This application demonstrates how historical sales patterns, calendar information, special events, and store/product characteristics can be used to generate daily demand forecasts.

Potential applications include:

- Supporting inventory planning and replenishment
- Identifying products with changing demand patterns
- Helping stores prepare for weekends and seasonal periods
- Providing a baseline for operational planning
- Comparing forecasting approaches using quantitative metrics

The forecasts are decision-support estimates and should be considered alongside business constraints and other operational information.

---

## 🔮 Future Improvements

- Expand the modeling workflow to more products and stores
- Evaluate additional forecasting algorithms
- Incorporate richer pricing and promotional features
- Improve recursive forecasting strategies
- Add prediction intervals or uncertainty estimates
- Monitor forecast errors as new sales data becomes available
- Develop more detailed product-level and store-level insights

---

## 👩‍💻 Author

Kasak