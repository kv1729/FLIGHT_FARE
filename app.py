import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved pipeline
loaded_pipeline = joblib.load(r'C:\Users\dream\OneDrive\Desktop\Flight_Fare_Prediction\MODEL\xgb_pipeline.pkl')

# Define categorical features
categorical_features = ['Journey_day', 'Airline', 'Class', 'Source', 'Departure', 'Total_stops',
                        'Arrival', 'Destination', 'On_weekend', 'Daytime_departure', 'Daytime_arrival']

# Function to get user input
def get_user_input():
    Journey_day = st.selectbox('Journey Day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    Airline = st.selectbox('Airline', ['Other', 'Indigo', 'GO FIRST', 'Air India', 'AirAsia', 'Vistara'])  # Replace with actual airline names
    Class = st.selectbox('Class', ['Economy', 'Premium Economy', 'Business', 'First'])
    Source = st.selectbox('Source', ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Kolkata', 'Chennai', 'Ahmedabad'])  # Replace with actual city names
    Departure = st.selectbox('Departure', ['After 6 PM', 'Before 6 AM', '12 PM - 6 PM', '6 AM - 12 PM'])
    Total_stops = st.selectbox('Total Stops', ['non-stop', '1-stop', '2+-stop'])
    Arrival = st.selectbox('Arrival', ['After 6 PM', 'Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM'])
    Destination = st.selectbox('Destination', ['Mumbai', 'Bangalore', 'Hyderabad', 'Kolkata', 'Chennai', 'Ahmedabad', 'Delhi'])  # Replace with actual city names
    On_weekend = st.selectbox('On Weekend', [False, True])
    Daytime_departure = st.selectbox('Daytime Departure', [False, True])
    Daytime_arrival = st.selectbox('Daytime Arrival', [False, True])
    Duration_in_hours = st.number_input('Duration in Hours', min_value=0.0, max_value=24.0, step=0.1)
    Days_left = st.number_input('Days Left', min_value=0, max_value=365, step=1)
    
    data = {
        'Journey_day': Journey_day,
        'Airline': Airline,
        'Class': Class,
        'Source': Source,
        'Departure': Departure,
        'Total_stops': Total_stops,
        'Arrival': Arrival,
        'Destination': Destination,
        'On_weekend': On_weekend,
        'Daytime_departure': Daytime_departure,
        'Daytime_arrival': Daytime_arrival,
        'Duration_in_hours': Duration_in_hours,
        'Days_left': Days_left
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Streamlit app main function
def main():
    st.title('Flight Price Prediction')
    
    st.write("Input the details of the flight:")
    input_df = get_user_input()
    
    # Make predictions
    if st.button('Predict'):
        prediction = loaded_pipeline.predict(input_df)
        st.subheader(f'Predicted Flight Price: {prediction[0]:.2f}')

if __name__ == '__main__':
    main()
