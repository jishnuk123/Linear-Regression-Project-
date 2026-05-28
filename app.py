import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the scaler
with open("scaler_california.pkl", "rb") as f:
    scaler = pickle.load(f)


#Graph 
y_all = np.load("y_all_california.npy")

# Recreate the model architecture and load weights
class LinearRegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(n_features=8)
model.load_state_dict(torch.load("model_california.pth"))
model.eval()




st.title("California House Price Predictor")
st.write("Adjust the sliders to predict a house price")

med_inc    = st.slider("Median Income (tens of thousands)", min_value=0.5,  max_value=15.0, value=5.0)
house_age  = st.slider("House Age (years)",                 min_value=1,    max_value=52,   value=20)
ave_rooms  = st.slider("Average Rooms",                     min_value=1.0,  max_value=10.0, value=5.0)
ave_bedrms = st.slider("Average Bedrooms",                  min_value=1.0,  max_value=5.0,  value=2.0)
population = st.slider("Neighborhood Population",           min_value=3,    max_value=3500, value=1000)
ave_occup  = st.slider("Average Occupancy",                 min_value=1.0,  max_value=10.0, value=3.0)
latitude   = st.slider("Latitude",                          min_value=32.5, max_value=42.0, value=37.0)
longitude  = st.slider("Longitude",                         min_value=-124.0, max_value=-114.0, value=-119.0)

# Predict button
if st.button("Predict Price"):
    # Scale the inputs
    X_input = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    X_scaled = scaler.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Get prediction
    with torch.no_grad():
        prediction = model(X_tensor).item()
    
    st.success(f"Predicted House Price: ${prediction * 100000:,.0f}")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(y_all, bins=40, color="#7C3AED", alpha=0.7, edgecolor="white")
    ax.axvline(prediction, color="#DC2626", linewidth=2.5, linestyle="--", label=f"Your prediction: ${prediction:,.0f}")
    ax.set_xlabel("House Price ($)")
    ax.set_ylabel("Count")
    ax.set_title("Where Your House Falls in the Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)