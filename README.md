# AQI Prediction Using Machine Learning

## Project Overview

The **AQI Prediction System** is a machine learning-based web application developed using Python and Streamlit to estimate the Air Quality Index (AQI) based on air pollution measurements.

The application uses a **Random Forest Regression** model to learn patterns from historical air quality data and predict AQI values. It provides an interactive interface where users can upload a dataset, explore air quality information, visualize data, evaluate model performance, and estimate AQI using pollutant measurements.

## Objectives

- To analyze air quality data using machine learning.
- To develop an AQI estimation model using Random Forest Regression.
- To visualize air quality trends and pollutant information.
- To evaluate the model using standard regression metrics.
- To provide a simple, interactive web interface for AQI estimation.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn

## Features

- Upload air quality datasets in CSV or supported ZIP format.
- Preview and explore dataset contents.
- Perform data preprocessing and handle missing values.
- Generate visualizations for air quality analysis.
- Train a Random Forest Regression model.
- Evaluate model performance using MAE, MSE, RMSE, and R² score.
- Compare actual and predicted AQI values.
- View feature importance.
- Enter pollutant measurements to estimate AQI.
- Download supported model and prediction result files.

## Dataset

The application is designed to work with a compatible air quality dataset, such as `city_day.csv`, containing historical AQI values and pollutant measurements.

The dataset must include the columns required by the application, including the target AQI column and the selected pollutant features.

## Machine Learning Algorithm

**Random Forest Regression** is used to estimate AQI. It combines predictions from multiple decision trees to produce a regression result.

The dataset is divided into training and testing sets. The model is trained using the training data and evaluated on unseen test data.

The evaluation metrics are:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Measures how well the model explains variation in the target values.

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Open the project folder

```bash
cd AQI-Prediction-ML
```

Navigate into the directory containing `app.py` and `requirements.txt` if the files are inside a nested folder.

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

### 5. Open the application

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

Upload a compatible dataset and follow the instructions displayed in the application.

## Project Limitations

- Prediction accuracy depends on dataset quality and the selected features.
- Results depend on the data used to train the model.
- This application estimates AQI from pollutant measurements; it does not automatically forecast future AQI values.

## Future Enhancements

- Integrate real-time air quality data from reliable APIs.
- Add future AQI forecasting.
- Improve model performance through hyperparameter tuning.
- Deploy the application online for public access.
- Add location-based air quality monitoring.

## Conclusion

The AQI Prediction System demonstrates how machine learning can be applied to air quality analysis and AQI estimation. By combining Python, Random Forest Regression, and Streamlit, the project provides an interactive platform for exploring pollution data and generating AQI estimates.

## Author

Developed as a machine learning project using Python and Streamlit.
