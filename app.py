import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

st.title("🌾 Crop Yield Prediction App")
st.write("This app predicts crop yield using Regression & Bagging Models.")
np.random.seed(42)
data = {
    "Rainfall": np.random.randint(200, 1200, 100),
    "Soil_pH": np.random.uniform(5.0, 8.0, 100),
    "Temperature": np.random.randint(15, 40, 100),
    "Fertilizer": np.random.randint(50, 300, 100),
}

df = pd.DataFrame(data)
df["Yield"] = (
    df["Rainfall"] * 0.05
    + df["Soil_pH"] * 10
    + df["Temperature"] * 2
    + df["Fertilizer"] * 0.3
    + np.random.normal(0, 10, 100)
)

st.dataframe(df.head())

X = df[["Rainfall", "Soil_pH", "Temperature", "Fertilizer"]]
y = df["Yield"]

st.header("⚙️ Train the Models")

if st.button("Train Models"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    bg = BaggingRegressor(n_estimators=20, random_state=42)
    bg.fit(X_train, y_train)
    bg_pred = bg.predict(X_test)

    st.success("Models trained successfully!")

    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    bg_mse = mean_squared_error(y_test, bg_pred)
    bg_r2 = r2_score(y_test, bg_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Linear Regression Performance")
        st.write(f"MSE: {lr_mse:.2f}")
        st.write(f"R²: {lr_r2:.3f}")

    with col2:
        st.subheader("📊 Bagging Regressor Performance")
        st.write(f"MSE: {bg_mse:.2f}")
        st.write(f"R²: {bg_r2:.3f}")

    st.info("Bagging Regressor is usually more stable for noisy datasets.")

st.header("🌱 Predict Crop Yield")

rain = st.slider("Rainfall (mm)", 200, 1200, 600)
ph = st.slider("Soil pH", 5.0, 8.0, 6.5)
temp = st.slider("Temperature (°C)", 15, 40, 25)
fert = st.slider("Fertilizer (kg)", 50, 300, 150)

if st.button("Predict Yield"):
    input_data = np.array([[rain, ph, temp, fert]])

    lr = LinearRegression().fit(X, y)
    bg = BaggingRegressor(n_estimators=20, random_state=42).fit(X, y)

    pred_lr = lr.predict(input_data)[0]
    pred_bg = bg.predict(input_data)[0]

    st.subheader("🔮 Prediction Results")
    st.success(f"🌿 Linear Regression: {pred_lr:.2f} kg/hectare")
    st.success(f"🌿 Bagging Regressor: {pred_bg:.2f} kg/hectare")
