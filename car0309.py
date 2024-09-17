import pandas as pd
import joblib
import streamlit as st
from PIL import Image

# Load the model and encoded columns
xgb_model = joblib.load('C:/Users/Darkk/OneDrive/Desktop/cars/xgboost_model.pkl')
encoded_columns = joblib.load('C:/Users/Darkk/OneDrive/Desktop/cars/encoded_columns.pkl')

# Define categorical columns
categorical_columns = ['Fuel Type', 'transmission', 'oem', 'Color', 'Location', 'RTO_grouped']

# Function to preprocess the input data
def preprocess_input(data, encoded_columns, categorical_columns):
    # Convert 'Seating Capacity' to integer
    data['Seating Capacity'] = data['Seating Capacity'].astype(int)
    
    # Convert categorical columns to dummy variables
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    data_encoded = data_encoded.reindex(columns=encoded_columns, fill_value=0)
    return data_encoded

# Function to predict car price
def predict_price(input_data):
    processed_input = preprocess_input(input_data, encoded_columns, categorical_columns)
    prediction = xgb_model.predict(processed_input)
    return prediction[0]

# Function to format price in INR format
def format_inr(number):
    s, *d = str(number).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)

# Main function for Streamlit app
def main():
    # Set background color and style
    st.markdown(
        """
        <style>
        .stApp {
            background-color:#D2B48C;
        }
        .logo {
            width: 80px; /* Adjust the size as needed */
            vertical-align: middle;
        }
        .title {
            display: flex;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the logo
    logo_path = "C://Users/Darkk/OneDrive/Desktop/cars/logo.jpg"
    logo = Image.open(logo_path)
    st.image(logo, width=80)

    # Main title
    st.title("Car Dheko: Used Car Price Predictor")

    st.sidebar.header('Enter car details:')

    # Input fields for user
    selected_brand = st.sidebar.selectbox('Brand', ['Audi', 'BMW', 'Chevrolet', 'Citroen', 'Datsun', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Land Rover', 'Lexus', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'MG', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'])
    color = st.sidebar.selectbox('Color', ['Silver', 'Black', 'White', 'Grey', 'Red', 'Blue', 'Green', 'Yellow', 'Brown'])
    transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
    fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric', 'Hybrid'])
    owner_no = st.sidebar.selectbox('Owner Number', ['0', '1', '2', '3', '4', '5'])
    location = st.sidebar.selectbox('Location', ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata'])
    rto_grouped = st.sidebar.selectbox('RTO Group', ['Andhra Pradesh', 'Delhi', 'Gujarat', 'Haryana', 'Karnataka', 'Maharashtra', 'Odisha', 'Puducherry', 'Rajasthan', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal'])
    model_year = st.sidebar.number_input('Model Year', min_value=1991, max_value=2024)
    seating_capacity = st.sidebar.selectbox('Seating Capacity', ['2', '4', '5', '6', '7', '8', '9', '10'])
    km = st.sidebar.number_input('Kms driven', min_value=0)
    engine = st.sidebar.number_input('Engine CC', min_value=0)

    # Prepare the input data
    input_data = pd.DataFrame({
        'oem': [selected_brand],
        'Color': [color],
        'Fuel Type': [fuel_type],
        'transmission': [transmission],
        'OwnerNo': [owner_no],
        'Location': [location],
        'RTO_grouped': [rto_grouped],
        'modelYear': [model_year],
        'km': [km],
        'Engine': [engine],
        'Seating Capacity': [seating_capacity]  # Seating Capacity remains as an integer
    })

    if st.sidebar.button('Predict'):
        prediction = predict_price(input_data)
        formatted_price = format_inr(prediction)
        
        # Display the prediction image
        prediction_image_path = "C:/Users/Darkk/OneDrive/Desktop/cars/car2.png"
        prediction_image = Image.open(prediction_image_path)
        st.image(prediction_image, caption='Predicted Price for Your Car', use_column_width=True)
        
        st.write(f'### The predicted price is: ₹ {formatted_price}')

if __name__ == '__main__':
    main()
