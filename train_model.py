import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv('Coca-Cola_stock_history.csv')

# Normalize column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Ensure required columns exist
required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

# -------------------------------
# 2. Convert Date Column
# -------------------------------
df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True, errors='coerce')
df.dropna(subset=['date'], inplace=True)

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# -------------------------------
# 3. Feature Engineering
# -------------------------------
df['daily_change'] = df['close'] - df['open']
df['daily_pct_change'] = (df['daily_change'] / df['open']) * 100
df['volatility'] = (df['high'] - df['low']) / df['open'] * 100
df['ma_5'] = df['close'].rolling(window=5).mean()
df['ma_20'] = df['close'].rolling(window=20).mean()
df['ma_50'] = df['close'].rolling(window=50).mean()

# Drop rows with NaNs introduced by rolling calculations
df.dropna(inplace=True)

# -------------------------------
# 4. Define Features and Target
# -------------------------------
features = [
    'open', 'high', 'low', 'volume',
    'daily_pct_change', 'volatility',
    'ma_5', 'ma_20', 'ma_50'
]
target = 'close'

X = df[features]
y = df[target]

# -------------------------------
# 5. Scale Features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 6. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# -------------------------------
# 7. Train Random Forest Model
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 8. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✅ R² Score: {r2:.4f}")
print(f"✅ RMSE: {rmse:.4f}")

# -------------------------------
# 9. Save Outputs
# -------------------------------
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
df.to_csv('Cleaned_Coca_Cola_stock_history.csv', index=False)

print("✅ Model, scaler, and cleaned dataset saved successfully.")
