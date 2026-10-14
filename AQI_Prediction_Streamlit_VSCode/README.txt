# AQI Prediction Streamlit Project

## Files
- `app.py` — complete Streamlit web application
- `requirements.txt` — Python packages needed

## Run in VS Code (Windows)
1. Install Python 3.10–3.13 and VS Code if they are not installed.
2. Extract this project ZIP to a folder, for example `C:\Users\lenovo\Documents\AQI_Streamlit_Project`.
3. Open VS Code.
4. Choose **File → Open Folder** and select the extracted folder.
5. Choose **Terminal → New Terminal**.
6. Install the packages:
   ```powershell
   py -m pip install -r requirements.txt
   ```
7. Start the app:
   ```powershell
   py -m streamlit run app.py
   ```
8. Open the Local URL printed in the terminal, usually `http://localhost:8501`.
9. Upload `archive.zip` (containing `city_day.csv`) or upload `city_day.csv` directly.

## Notes
- The app uses Random Forest Regression and the notebook's pollutant feature list.
- Default model settings match the notebook: 80:20 split, 150 trees, max depth 20, minimum leaf size 2, random state 42.
- It estimates AQI from same-observation pollutant readings; it is not a future-day forecast.
