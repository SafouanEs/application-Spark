
import streamlit as st
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

# Initialize a Spark session
spark = SparkSession.builder.appName("Fraud Detection").getOrCreate()

def initialize_model(model_path):
    # Load the trained model
    model = PipelineModel.load(model_path)
    return model

def predict_fraud(model, input_data):
    # Define schema for the input data
    schema = StructType([
        StructField("Amount", DoubleType(), True),
        StructField("Transaction_Type_Index", DoubleType(), True),
        StructField("Location_Index", DoubleType(), True),
        StructField("Merchant_Index", DoubleType(), True),
        StructField("Year", DoubleType(), True),
        StructField("Month", DoubleType(), True),
        StructField("Day", DoubleType(), True),
        StructField("HourOfDay", DoubleType(), True),
    ])
    
    # Convert input data to Spark DataFrame
    data_df = spark.createDataFrame([input_data], schema=schema)

    # Make predictions
    predictions = model.transform(data_df)
    return predictions

def main():
    st.title('Fraud Detection Application')

    model_path = st.sidebar.text_input("Enter the path to your model")

    amount = st.number_input('Transaction Amount', min_value=0.0, format="%.2f")
    transaction_type_index = st.number_input('Transaction Type Index', min_value=0.0, format="%.2f")
    location_index = st.number_input('Location Index', min_value=0.0, format="%.2f")
    merchant_index = st.number_input('Merchant Index', min_value=0.0, format="%.2f")
    year = st.number_input('Year', min_value=2000, max_value=2100, step=1)
    month = st.number_input('Month', min_value=1, max_value=12, step=1)
    day = st.number_input('Day of Week', min_value=1, max_value=7, step=1)
    hour_of_day = st.number_input('Hour of Day', min_value=0, max_value=23, step=1)

    if st.button('Predict Fraud'):
        if model_path:
            try:
                # Initialize and load the model
                model = initialize_model(model_path)

                # Prepare input data
                input_data = (amount, transaction_type_index, location_index, merchant_index, year, month, day, hour_of_day)

                # Get predictions
                prediction = predict_fraud(model, input_data)

                # Extract and display the prediction result
                prediction_label = prediction.select('prediction').first()[0]
                st.write(f"The transaction is predicted to be: {'Fraudulent' if prediction_label == 1 else 'Not Fraudulent'}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please provide the path to your model.")

if __name__ == "__main__":
    main()
