import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

# -----------------------------
# Page configuration & CSS
# -----------------------------
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-container {
        padding: 2rem;
    }
    .section-title {
        font-size: 2rem;
        color: #2c3e50;
        border-bottom: 2px solid #bdc3c7;
        padding-bottom: 0.5rem;
    }
    .sub-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='section-title'>Personal Fitness Tracker</h1>", unsafe_allow_html=True)
st.write("Enter your details on the sidebar and click **Submit** to view your personalized prediction and suggestions.")

# -----------------------------
# Sidebar - Input Form
# -----------------------------
with st.sidebar.form("input_form"):
    st.header("Input Parameters")
    age = st.slider("Age", 10, 100, 30, help="Your age in years")
    bmi = st.slider("BMI", 15, 40, 20, help="Body Mass Index")
    duration = st.slider("Workout Duration (min)", 0, 35, 15, help="Duration of your exercise session")
    heart_rate = st.slider("Heart Rate", 60, 130, 80, help="Average heart rate during exercise")
    body_temp = st.slider("Body Temperature (°C)", 36, 42, 38, help="Your body temperature during exercise")
    gender_button = st.radio("Gender", ("Male", "Female"))
    
    st.markdown("### Additional Details")
    daily_steps = st.slider("Daily Steps", 0, 20000, 5000, help="Your average number of steps per day")
    sleep_duration = st.slider("Sleep Duration (hrs)", 0, 12, 7, help="Average hours of sleep per night")
    water_intake = st.slider("Water Intake (liters)", 0.0, 5.0, 2.0, step=0.1, help="Your average daily water intake")
    
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    # Convert gender to numerical
    gender = 1 if gender_button == "Male" else 0

    # Create user input DataFrame (preserve all columns for suggestions)
    original_user_input = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender],
        "Daily_Steps": [daily_steps],
        "Sleep_Duration": [sleep_duration],
        "Water_Intake": [water_intake]
    })

    # -----------------------------
    # Display Input Parameters
    # -----------------------------
    st.markdown("<div class='sub-section'>", unsafe_allow_html=True)
    st.subheader("Your Input Parameters")
    st.write(original_user_input)
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Load and Preprocess Data
    # -----------------------------
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")

    # Merge datasets on User_ID and drop User_ID column
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)

    # Train/test split (for model training)
    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    # Add BMI column to training data
    for data in [exercise_train_data, exercise_test_data]:
        data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
        data["BMI"] = round(data["BMI"], 2)

    # Use the original six features for model training (adjust if you have additional columns)
    features_cols = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
    exercise_train_data = exercise_train_data[features_cols]
    exercise_test_data = exercise_test_data[features_cols]

    # One-hot encode for gender
    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]

    # -----------------------------
    # Train Model
    # -----------------------------
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)

    # Prepare prediction input (reindex to model features)
    user_input_pred = original_user_input.reindex(columns=X_train.columns, fill_value=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = random_reg.predict(user_input_pred)
    st.markdown("<div class='sub-section'>", unsafe_allow_html=True)
    st.subheader("Predicted Calories Burned")
    st.write(f"**{round(prediction[0], 2)} kilocalories**")
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Similar Results
    # -----------------------------
    st.markdown("<div class='sub-section'>", unsafe_allow_html=True)
    st.subheader("Similar Results")
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) &
                               (exercise_df["Calories"] <= calorie_range[1])]
    st.write(similar_data.sample(5))
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Personalized Suggestions
    # -----------------------------
    def generate_suggestions(input_df):
        suggestions = []
        age = input_df["Age"].values[0]
        duration = input_df["Duration"].values[0]
        heart_rate = input_df["Heart_Rate"].values[0]
        daily_steps = input_df["Daily_Steps"].values[0]
        sleep_duration = input_df["Sleep_Duration"].values[0]
        water_intake = input_df["Water_Intake"].values[0]
        bmi = input_df["BMI"].values[0]

        if duration < 20:
            suggestions.append("Increase your workout duration to boost calorie burn.")
        else:
            suggestions.append("Great job on maintaining a good workout duration!")
        
        if daily_steps < 7000:
            suggestions.append("Try to add more daily steps for improved cardiovascular health.")
        else:
            suggestions.append("Your daily steps are impressive!")
        
        if sleep_duration < 7:
            suggestions.append("Increase your sleep duration for better recovery.")
        else:
            suggestions.append("You have a good sleep duration; keep it up!")
        
        if water_intake < 2:
            suggestions.append("Increase your water intake to stay optimally hydrated.")
        else:
            suggestions.append("Your water intake is on track!")
        
        if heart_rate < 80:
            suggestions.append("A slightly higher heart rate during workouts might burn more calories.")
        else:
            suggestions.append("Your heart rate indicates a good workout intensity.")
        
        if bmi > 25:
            suggestions.append("Incorporate cardio and strength training to better manage your BMI.")
        else:
            suggestions.append("Your BMI is within a healthy range!")
        
        return suggestions

    suggestions = generate_suggestions(original_user_input)
    
    st.markdown("<div class='sub-section'>", unsafe_allow_html=True)
    st.subheader("Personalized Suggestions")
    cols = st.columns(2)
    half = len(suggestions) // 2
    with cols[0]:
        for s in suggestions[:half]:
            st.write("- " + s)
    with cols[1]:
        for s in suggestions[half:]:
            st.write("- " + s)
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # General Information
    # -----------------------------
    st.markdown("<div class='sub-section'>", unsafe_allow_html=True)
    st.subheader("General Information")
    boolean_age = (exercise_df["Age"] < original_user_input["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < original_user_input["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < original_user_input["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < original_user_input["Heart_Rate"].values[0]).tolist()

    st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
    st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
    st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
    st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
