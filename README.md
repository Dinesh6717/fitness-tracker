Personal Fitness Tracker  

Personal Fitness Tracker is a Streamlit-based web application that predicts calorie burn based on user fitness data. It provides personalized insights and fitness recommendations using machine learning.  


 Features  

- User-friendly interface with interactive sidebar  
- Calorie prediction model using Random Forest Regressor  
- Personalized fitness recommendations based on user input  
- Comparison with similar users for better insights  
- Data visualization for fitness trends  

 Project Structure  

Personal-Fitness-Tracker
│── app.py            # Main Streamlit application
│── calories.csv      # Dataset containing calorie burn data
│── exercise.csv      # Dataset containing exercise-related data
│── README.md         # Project documentation


Installation & Setup  

1. Clone the Repository
   
git clone https://github.com/Dinesh6717/fitness-tracker
cd Personal-Fitness-Tracker

 2. Install Dependencies  
pip install -r requirements.txt

3. Run the Application  

streamlit run app.py

 Technologies Used  

- Python 3.8+ – Main programming language  
- Streamlit – Web application framework  
- scikit-learn – Machine learning model training  
- pandas & NumPy – Data manipulation and preprocessing  
- Matplotlib & Seaborn – Data visualization  

Dataset Used  

The project uses real-world fitness datasets:  

1. `calories.csv` – Contains calorie burn data for different users.  
2. `exercise.csv` – Includes user fitness details such as heart rate, duration, and body temperature.  

These datasets are merged and preprocessed before training the Random Forest Regressor model.  

 How It Works  

1. User Input – Enter age, BMI, workout duration, heart rate, sleep duration, and daily steps in the sidebar.  
2. Machine Learning Model – Predicts calories burned using Random Forest Regressor.  
3. Personalized Suggestions – Offers fitness advice based on user input.  
4. Comparison with Others – Displays similar users' fitness data for reference.  

Future Improvements  

- Improve model accuracy by adding more fitness parameters.  
- Implement a deep learning model for better calorie predictions.  
- Allow users to track their progress over time.  
- Add Firebase integration to store and retrieve user data.  

Contributing  

Contributions are welcome! If you’d like to improve the project:  

1. Fork the repository  
2. Create a new branch   
3. Commit your changes  
4. Push the branch  
5. Open a Pull Request  

