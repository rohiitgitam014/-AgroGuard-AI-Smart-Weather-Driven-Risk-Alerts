import streamlit as st
import pandas as pd
import requests
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="🌾 AgroGuard AI", layout="centered")
st.title("🌾 AgroGuard AI: Smart Weather-Driven Risk Alerts")
st.markdown("Get real-time agricultural risk alerts based on live weather conditions using NASA POWER data and machine learning.")

# --- Get Real-time Weather Data from NASA API ---
def get_weather_data(lat, lon):
    yesterday = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,WS2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={yesterday}&end={yesterday}&format=JSON"

    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()["properties"]["parameter"]
        weather = {
            "T2M": list(data["T2M"].values())[0:],
            "RH2M": list(data["RH2M"].values())[0:],
            "WS2M": list(data["WS2M"].values())[0:],
            "PRECTOTCORR": list(data["PRECTOTCORR"].values())[0],
        }
        return pd.DataFrame([weather])
    else:
        st.error("Failed to fetch data from NASA POWER API.")
        return None

# --- Sample Historical Dataset with Labels ---
def get_historical_data():
    data = {
        "T2M": [38.5, 35.0, 29.5, 33.0, 41.2, 28.7],  # Temperature in Celsius
        "RH2M": [35, 60, 85, 70, 25, 90],             # Relative Humidity in %
        "WS2M": [3.0, 1.5, 0.8, 2.5, 4.2, 1.0],        # Wind Speed at 2m in m/s
        "PRECTOTCORR": [0.0, 1.2, 12.5, 5.0, 0.0, 15.8],  # Precipitation in mm
        "Risk_Level": ["Low", "Moderate", "High", "Moderate", "Low", "High"]
    }
    return pd.DataFrame(data)
# --- Train the ML Model and Return Evaluation Metrics ---
def train_model():
    df = get_historical_data()
    label_encoder = LabelEncoder()
    df['Risk_Label'] = label_encoder.fit_transform(df['Risk_Level'])

    X = df[["T2M", "RH2M", "WS2M", "PRECTOTCORR"]]
    y = df["Risk_Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Dynamically extract only present labels
    present_labels = sorted(set(y_test) | set(y_pred))
    target_names = label_encoder.inverse_transform(present_labels)

    report = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=target_names,
        output_dict=True
    )
    report = pd.DataFrame(report).transpose()
    matrix = confusion_matrix(y_test, y_pred, labels=present_labels)

    return model, label_encoder, report, matrix


# --- User Input for Location ---
st.sidebar.header("📍 Enter Location")
lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.4f")

# --- Train model ---
model, label_encoder, report, matrix = train_model()

# --- Get real-time weather data ---
if st.sidebar.button("🚀 Fetch and Predict Risk Level"):
    weather_df = get_weather_data(lat, lon)
    if weather_df is not None:
        st.subheader("📊 Real-time Weather Data")
        st.dataframe(weather_df)

        prediction = model.predict(weather_df)[0]
        predicted_risk = label_encoder.inverse_transform([prediction])[0]

        st.subheader("⚠️ Predicted Risk Level")
        st.success(f"🌡️ Based on current weather, the predicted risk level is: **{predicted_risk}**")

# --- Evaluation Section ---
st.subheader("🧪 Model Evaluation on Real time Data")

# Classification Report
st.markdown("**Classification Report:**")
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.background_gradient(cmap='Blues'))
