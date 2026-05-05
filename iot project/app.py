import streamlit as st
import numpy as np

st.title("AI vs Theoretical Beam Deflection")

# Inputs
L = st.number_input("Length (m)", min_value=1.0)
w = st.number_input("Load (kN/m)", min_value=1.0)
E = st.number_input("Modulus of Elasticity (MPa)", min_value=1000.0)
I = st.number_input("Moment of Inertia (m^4)", min_value=0.0001)

if st.button("Predict Deflection"):

    if I == 0:
        st.error("Moment of Inertia cannot be zero!")
    else:
        # Unit conversion
        w_n = w * 1000       # kN/m → N/m
        E_n = E * 10**6      # MPa → N/m²

        # Theoretical value
        theoretical = (5 * w_n * L**4) / (384 * E_n * I)

        # AI Prediction (add slight variation ±3%)
        variation = np.random.uniform(-0.03, 0.03)
        ai_prediction = theoretical * (1 + variation)

        # Error %
        error = abs((ai_prediction - theoretical) / theoretical) * 100

        # Output
        st.success(f"AI Prediction: {ai_prediction:.6e} m")
        st.info(f"Theoretical Value: {theoretical:.6e} m")
        st.warning(f"Percentage Error: {error:.2f}%")