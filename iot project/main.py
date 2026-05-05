# beam_deflection_ai.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# -------------------------------
# STEP 1: Generate Dataset
# -------------------------------

data = []

for i in range(1000):
    L = np.random.uniform(2, 10)  # Length (m)
    w = np.random.uniform(5, 50)  # Load (kN/m)
    E = np.random.uniform(20000, 35000) * 10**6  # Convert MPa to N/m²
    I = np.random.uniform(0.0001, 0.01)  # Moment of Inertia (m^4)

    # Deflection formula
    delta = (5 * w * L**4) / (384 * E * I)

    data.append([L, w, E, I, delta])

# Create DataFrame
df = pd.DataFrame(data, columns=["Length", "Load", "E", "I", "Deflection"])

# Save dataset
df.to_csv("beam_data.csv", index=False)
print("Dataset saved as /tmp/beam_data.csv")

# -------------------------------
# STEP 2: Prepare Data
# -------------------------------

X = df[["Length", "Load", "E", "I"]]
y = df["Deflection"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 3: Train Model
# -------------------------------

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# STEP 4: Evaluate Model
# -------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("MAE:", mae)
print("R2 Score:", r2)

# -------------------------------
# STEP 5: Save Model
# -------------------------------

joblib.dump(model, "beam_model.pkl")
print("Model saved as /tmp/beam_model.pkl")

# -------------------------------
# STEP 6: Test Prediction
# -------------------------------

# Example input: [Length, Load, E, I]
sample = [[5, 20, 25000 * 10**6, 0.002]]

predicted_deflection = model.predict(sample)

print("\nSample Prediction:")
print("Predicted Deflection:", predicted_deflection[0])

