# AQI Prediction Using Machine Learning

## Project Overview
The AQI Prediction System is a machine learning web application developed using Python and Streamlit. It estimates the Air Quality Index (AQI) based on air pollution measurements using a Random Forest Regression model.

The application provides an interactive interface for uploading air quality datasets, exploring data, visualizing pollution information, evaluating model performance, and estimating AQI values.

## Objectives
- Analyze air quality data using machine learning.
- Develop an AQI estimation model using Random Forest Regression.
- Visualize pollutant measurements and air quality patterns.
- Evaluate model performance using regression metrics.
- Provide a simple web interface for AQI estimation.

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Main Features
- Upload a compatible CSV or ZIP dataset.
- Preview and analyze air quality data.
- Handle missing values during preprocessing.
- Visualize dataset patterns.
- Train a Random Forest Regression model.
- Evaluate performance using MAE, MSE, RMSE, and R² score.
- View actual versus predicted values and feature importance.
- Estimate AQI from pollutant measurements.
- Download supported model and result files.

## Dataset
The application is designed to use a compatible air quality dataset, such as `city_day.csv`, containing an AQI column and pollutant measurements. The required columns depend on the features available in the dataset.

## How to Run the Project

1. Install Python and Git.
2. Clone this repository.
3. Open the folder containing `app.py` and `requirements.txt`.
4. Install the dependencies:

   `python -m pip install -r requirements.txt`

5. Start the application:

   `python -m streamlit run app.py`

6. Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Model Evaluation
The application uses standard regression metrics:
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Measures how well the model explains variation in AQI values.

## Limitations
The prediction depends on dataset quality and the pollutant measurements supplied. The application estimates AQI for the supplied observation; it does not automatically forecast AQI for a future day.

## Future Enhancements
- Integrate real-time air quality APIs.
- Add future AQI forecasting.
- Improve model performance through hyperparameter tuning.
- Deploy the application online.

## Conclusion
This project demonstrates the use of machine learning for air quality analysis and AQI estimation through an interactive Streamlit web application.
