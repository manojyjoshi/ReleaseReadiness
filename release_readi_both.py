import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the dataset to get feature names for input fields
df = pd.read_csv('release_readiness_dataset.csv')

# Split the dataset into features (X) and target (y)
X = df.drop('Release_Readiness_Score', axis=1)  # Features
y = df['Release_Readiness_Score']  # Target

# Streamlit App Title
st.markdown("<h5 style='margin-top: 0px; padding-top: 0px; margin-bottom: 0px;'>Release readiness score predictions</h5>", unsafe_allow_html=True)
# st.title("Release Readiness Score Prediction")
# Use Markdown for custom title styling with reduced spacing
# st.markdown("<h4 style='margin-top: 0px; padding-top: 0px; margin-bottom: 5px;'>Sample Data - Good and Bad Release Readiness</h3>", unsafe_allow_html=True)

# Step 1: Display Sample Data (Good and Bad Release Readiness)
sample_data = df[['Release_Readiness_Score'] + list(X.columns)].sample(10, random_state=42)

# Display the samples with corresponding labels for "good" and "bad" release readiness
good_sample = sample_data[sample_data['Release_Readiness_Score'] >= 0.8]
bad_sample = sample_data[sample_data['Release_Readiness_Score'] < 0.4]

# Show good release readiness samples
st.markdown("<h5 style='margin-top: 0px; padding-top: 0px; margin-bottom: 0px;'>Good Release Readiness Examples (Score ≥ 0.8)</h5>", unsafe_allow_html=True)
# st.subheader("Good Release Readiness Examples (Score ≥ 0.8)")
if not good_sample.empty:
    st.dataframe(good_sample.head(2))  # Display only the first 2 rows
else:
    st.write("")

# Now show bad examples with a score < 0.4
st.markdown("<h5 style='margin-top: 0px; padding-top: 0px; margin-bottom: 0px;'>Bad Release Readiness Examples (Score < 0.4)</h5>", unsafe_allow_html=True)
# st.subheader("Bad Release Readiness Examples (Score < 0.4)")
if not bad_sample.empty:
    st.dataframe(bad_sample.head(2))  # Display only the first 2 rows
else:
    # Artificial bad data (now converted to percentage format for consistency)
    artificial_bad = pd.DataFrame({
        'Release_Readiness_Score': [35, 20],  # Convert these to percentages (35%, 20%, 10%)
        X.columns[0]: [200, 300],  # Feature 1: High number of defects (integer)
        X.columns[1]: [50, 100],  # Feature 2: High number of open issues (integer)
        X.columns[2]: [10, 15],  # Feature 3: Low test coverage (percentage)
        X.columns[3]: [15, 25],  # Feature 4: Long cycle time (integer)
        X.columns[4]: [5,7],  # Feature 5: Low severity issues (integer)
        X.columns[5]: [90, 80],  # Feature 6: High technical debt (percentage)
        X.columns[6]: [5, 10],  # Feature 7: Poor code quality (percentage)
        X.columns[7]: [50, 60],  # Feature 8: Low test automation percentage (percentage)
        X.columns[8]: [20, 15],  # Feature 9: High production incidents (integer)
        X.columns[9]: [100, 150],  # Feature 10: High number of support tickets (integer)
        X.columns[10]: [30, 40],  # Feature 11: High number of manual testing hours (integer)
    })
    st.write("")
    st.dataframe(artificial_bad)  # Now properly formatted with percentage-like values

# Adjust layout for reduced scrolling and better organization
col1 = st.columns([1])[0]

# Column 1: Input fields and predicted score
with col1:
    st.markdown(
        "<h5 style='margin-top: 0px; padding-top: 0px; margin-bottom: 5px;'>Input Parameters</h5>",
        unsafe_allow_html=True)
    # st.header("Input Parameters")

    # Set columns for input fields (5 fields per row)
    input_cols = st.columns(5)  # 5 fields in a row
    input_data = {}

    # Create input fields with proper data type detection
    for i, column in enumerate(X.columns):
        # Checking if the column name suggests it's related to defects/issues (likely integer values)
        if any(keyword in column.lower() for keyword in ['defect', 'issue', 'ticket', 'incident', 'hour']):
            # Integer fields (e.g., defect counts)
            input_data[column] = input_cols[i % 5].number_input(f"{column}", value=0, step=1, key=column)
        else:
            # Float fields (e.g., percentages)
            input_data[column] = input_cols[i % 5].number_input(f"{column}", value=0.0, format="%.2f", step=0.01, key=column)

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Feature scaling
    input_scaled = scaler.transform(input_df)

    # Predict button
    if st.button("Predict"):
        # Make the prediction
        prediction = model.predict(input_scaled)

        # Display the prediction result
        st.subheader(f"Predicted Release Readiness Score: {prediction[0]:.2f}")
