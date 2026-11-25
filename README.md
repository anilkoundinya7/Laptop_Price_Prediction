
# Laptop Price Prediction using Machine Learning

## Project Overview
This project predicts laptop prices based on hardware specifications using Machine Learning.

## Dataset
- Dataset: Laptop Price Dataset
- Records: 1303
- Target Variable: Price_euros

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Workflow
1. Data Loading
2. Exploratory Data Analysis
3. Data Preprocessing
4. Feature Scaling using MinMaxScaler
5. Categorical Encoding using OneHotEncoder
6. Train-Test Split
7. Random Forest Regression
8. Model Evaluation

## Model
Random Forest Regressor

## Evaluation Metrics
- MAE
- MSE
- RMSE
- R2 Score

## Streamlit Application
The project includes a Streamlit web application where users can enter laptop specifications and receive an estimated laptop price.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset file `laptop_price.csv` in the project folder.

Train the model:

```bash
python laptop_price_prediction.py
```

Run the application:

```bash
python -m streamlit run app.py
```

## Author
Anil
