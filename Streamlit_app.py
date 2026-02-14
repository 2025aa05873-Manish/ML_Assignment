import streamlit as st
import pandas as pd
import numpy as np

# Title and description
st.title("My Streamlit App 🚀")
st.write("A simple demo app deployed from GitHub.")

# Sidebar for user input
st.sidebar.header("User Input")
user_name = st.sidebar.text_input("Enter your name:", "Guest")
num_points = st.sidebar.slider("Number of data points:", 10, 100, 50)

# Generate random data
data = pd.DataFrame({
    "x": np.arange(num_points),
    "y": np.random.randn(num_points).cumsum()
})

# Display greeting
st.subheader(f"Hello, {user_name}!")
st.write("Here’s a random dataset visualization:")

# Show dataframe
st.dataframe(data)

# Line chart
st.line_chart(data.set_index("x"))

# Add a checkbox for extra info
if st.checkbox("Show summary statistics"):
    st.write(data.describe())
