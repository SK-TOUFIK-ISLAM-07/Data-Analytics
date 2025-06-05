# Data-Analytics


This repository contains two independent data science projects applying machine learning, data analysis, and forecasting methods to real-world problems:

1. **📈 Coca-Cola Stock Analysis** — A time series forecasting and regression-based financial analysis project.
2. **🌍 Climate Change Modeling** — A sentiment and engagement prediction model based on social media comments about climate change.

---

## 📁 Projects Overview

### 📈 Coca-Cola Stock Analysis

**Difficulty**: Intermediate  
**Tools**: Jupyter Notebook, Python, Scikit-learn, TensorFlow, Matplotlib, Seaborn  
**Domain**: Finance, Time Series Forecasting

#### 📊 Dataset
- Historical stock data of Coca-Cola (Ticker: KO) from 1962 to present.
- Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits`
- Source: Yahoo Finance or equivalent  
- 📥 [Dataset Download Link](#) *(Replace with actual link)*

#### 🎯 Goals
- Analyze historical stock trends.
- Forecast closing prices using ML models.
- Evaluate return and risk via Sharpe Ratio, volatility, and cumulative returns.

#### 🧪 Features
- Daily % Change, Volatility, Moving Averages
- Cumulative Return, Lag Features
- Feature Scaling and Correlation Analysis

#### 🧠 Models
- **Linear Regression**
- **Random Forest Regressor**
- **LSTM Neural Network**

#### 📈 Evaluation
- RMSE, R² Score, Sharpe Ratio
- Visuals: Heatmaps, Moving Averages, Cumulative Return, Actual vs Predicted

#### 🔮 Future Work
- Add ARIMA/Prophet models
- Real-time stock prediction using API
- Web dashboard via Flask/Streamlit

---

### 🌍 Climate Change Modeling

**Difficulty**: Advanced  
**Tools**: Jupyter Notebook, VS Code  
**Technologies**: Machine Learning, Natural Language Processing (NLP)  
**Domain**: Social Media Analytics / Public Sentiment

#### 📊 Dataset
- 500+ user comments from NASA’s Facebook page (2020–2023)
- Columns:
  - `date`
  - `likesCount`
  - `profileName` (hashed)
  - `commentsCount`
  - `text`
- [Visit NASA Climate Page](https://web.facebook.com/NASAClimateChange/)

#### 🎯 Goals
- Perform sentiment analysis on climate-related comments.
- Predict user engagement (likes/replies).
- Identify trends in public opinion over time.

#### 🧪 Features
- Text cleaning and preprocessing
- Sentiment tagging using VADER
- TF-IDF vectorization, meta-features
- Word clouds and data visualizations

#### 🧠 Models

**Classification (Sentiment):**
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (Best Performer)
- MLP Classifier

**Regression (Engagement Prediction):**
- Random Forest Regressor
- Gradient Boosting Regressor
- MLP Regressor

#### 📈 Evaluation
- **Classification**: Accuracy, F1-Score, Confusion Matrix  
- **Regression**: MAE, MSE, R² Score  
- Visuals: Sentiment Timeline, Word Clouds, Residual Plots

#### 🔮 Future Work
- BERT-based sentiment classifier
- Integrate NOAA/IPCC datasets
- Public deployment with Flask or Django

---

## ⚖️ Ethical Considerations

- Climate change project anonymizes all user data using SHA-256 hashing.
- Coca-Cola stock data is publicly available.
- These projects are intended solely for **educational and research purposes** and do not constitute financial advice.

---

## 🙏 Acknowledgements

- 📊 Thanks to [Yahoo Finance](https://finance.yahoo.com) for financial data.
- 🌐 Gratitude to [NASA Climate Change](https://web.facebook.com/NASAClimateChange/) for public comment access.
- ❤️ Built with open-source libraries: Scikit-learn, Pandas, TensorFlow, Seaborn, Matplotlib, NLTK, and others.

---

