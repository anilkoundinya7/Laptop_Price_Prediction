
# ============================================================
# LAPTOP PRICE PREDICTOR - STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

h1 {
    color: #1f3c88;
    text-align: center;
}

.stButton>button {
    background-color: #1f3c88;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    artifacts = joblib.load("laptop_price_artifacts.pkl")

    return artifacts


artifacts = load_artifacts()

model = artifacts["model"]

scaler = artifacts["scaler"]

encoder = artifacts["encoder"]

num_cols = artifacts["num_cols"]

cat_cols = artifacts["cat_cols"]


# ============================================================
# 4. APPLICATION TITLE
# ============================================================

st.title("💻 LAPTOP PRICE PREDICTOR")

st.write(
    "Predict the estimated price of a laptop based on its specifications."
)

st.divider()


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title("About the Project")

st.sidebar.write(
    "This application uses Machine Learning "
    "to predict laptop prices."
)

st.sidebar.subheader("Machine Learning Model")

st.sidebar.write("Model: Random Forest Regressor")

st.sidebar.write("Numerical Scaling: MinMaxScaler")

st.sidebar.write("Categorical Encoding: OneHotEncoder")


# ============================================================
# 6. USER INPUTS
# ============================================================

st.subheader("Enter Laptop Specifications")

col1, col2 = st.columns(2)


with col1:

    company = st.selectbox(
        "Laptop Company",
        [
            "Apple", "HP", "Acer", "Asus", "Dell",
            "Lenovo", "Chuwi", "MSI", "Microsoft",
            "Toshiba", "Huawei", "Xiaomi", "Vero",
            "Razer", "Mediacom", "Samsung", "Google",
            "Fujitsu", "LG"
        ]
    )

    typename = st.selectbox(
        "Laptop Type",
        [
            "Ultrabook",
            "Notebook",
            "Netbook",
            "Gaming",
            "2 in 1 Convertible",
            "Workstation"
        ]
    )

    inches = st.selectbox(
        "Screen Size (Inches)",
        [11.6, 12.0, 13.3, 14.0, 15.6, 17.3]
    )

    screenresolution = st.selectbox(
        "Screen Resolution",
        [
            "1366x768",
            "1600x900",
            "1920x1080",
            "2560x1440",
            "3840x2160",
            "1920x1200",
            "2560x1600",
            "3200x1800",
            "2880x1800"
        ]
    )

    cpu = st.selectbox(
        "CPU",
        [
            "Intel Core i3 6006U 2GHz",
            "Intel Core i5 7200U 2.5GHz",
            "Intel Core i7 7700HQ 2.8GHz",
            "Intel Core i5 8250U 1.6GHz",
            "Intel Core i7 8550U 1.8GHz",
            "Intel Core i7 8750H 2.2GHz",
            "AMD A9-Series 9420 3GHz",
            "AMD Ryzen 5 2500U 2GHz"
        ]
    )


with col2:

    ram = st.selectbox(
        "RAM (GB)",
        [2, 4, 6, 8, 12, 16, 24, 32, 64]
    )

    memory = st.selectbox(
        "Storage",
        [
            "128GB SSD",
            "256GB SSD",
            "512GB SSD",
            "1TB SSD",
            "128GB HDD",
            "500GB HDD",
            "1TB HDD",
            "256GB SSD + 1TB HDD",
            "128GB SSD + 1TB HDD",
            "256GB SSD + 2TB HDD",
            "32GB Flash Storage",
            "64GB Flash Storage"
        ]
    )

    gpu = st.selectbox(
        "GPU",
        [
            "Intel HD Graphics 620",
            "Intel UHD Graphics 620",
            "Nvidia GeForce GTX 1050",
            "Nvidia GeForce GTX 1060",
            "Nvidia GeForce GTX 1070",
            "Nvidia GeForce 940MX",
            "AMD Radeon R5 M330",
            "AMD Radeon 530",
            "Intel HD Graphics 400"
        ]
    )

    opsys = st.selectbox(
        "Operating System",
        [
            "Windows 10",
            "Windows 7",
            "Windows 10 S",
            "macOS",
            "Mac OS X",
            "Linux",
            "No OS",
            "Chrome OS",
            "Android"
        ]
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1
    )


# ============================================================
# 7. PREDICTION BUTTON
# ============================================================

st.divider()

if st.button("PREDICT LAPTOP PRICE"):

    # Create input dictionary

    input_data = {

        "Company": company,

        "TypeName": typename,

        "Inches": inches,

        "ScreenResolution": screenresolution,

        "Cpu": cpu,

        "Ram": ram,

        "Memory": memory,

        "Gpu": gpu,

        "OpSys": opsys,

        "Weight": weight

    }

    # Convert dictionary into DataFrame

    input_df = pd.DataFrame([input_data])

    # Make sure columns match training feature names

    input_df = input_df[num_cols + cat_cols]

    # Numerical transformation

    input_num = scaler.transform(
        input_df[num_cols]
    )

    # Categorical transformation

    input_cat = encoder.transform(
        input_df[cat_cols]
    ).toarray()

    # Combine transformed data

    final_input = np.hstack(
        (input_num, input_cat)
    )

    # Predict laptop price

    prediction = model.predict(final_input)

    # Display prediction

    st.success("Laptop Price Predicted Successfully!")

    st.subheader("Estimated Laptop Price")

    st.metric(
        label="Predicted Price",
        value=f"€ {prediction[0]:,.2f}"
    )

    st.info(
        "This is an estimated price generated by the "
        "Random Forest Machine Learning model."
    )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "Laptop Price Prediction | Machine Learning Project"
)
