import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("Used Car Price Prediction using SVR")

df = pd.read_csv("final_indian_used_car_market_dataset.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encoding
label_encoder = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

# Features & target
X = df.drop('listed_price', axis=1)
y = df['listed_price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVR Model
model = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display results
st.subheader("Model Evaluation")
st.write("MAE:", mae)
st.write("MSE:", mse)
st.write("RMSE:", rmse)
st.write("R2 Score:", r2)