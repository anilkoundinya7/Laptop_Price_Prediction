
# ============================================================
# LAPTOP PRICE PREDICTION - COMPLETE ML PROJECT
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("laptop_price.csv", encoding="latin1")

print("\nDATASET LOADED SUCCESSFULLY")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

# Graph 1: Laptop Price Distribution

plt.figure(figsize=(8, 5))
sns.histplot(df["Price_euros"], kde=True)
plt.title("Laptop Price Distribution")
plt.xlabel("Price in Euros")
plt.ylabel("Number of Laptops")
plt.tight_layout()
plt.show()


# Graph 2: Company vs Average Price

plt.figure(figsize=(12, 6))

company_price = (
    df.groupby("Company")["Price_euros"]
    .mean()
    .sort_values(ascending=False)
)

sns.barplot(x=company_price.index, y=company_price.values)

plt.title("Average Laptop Price by Company")
plt.xlabel("Company")
plt.ylabel("Average Price in Euros")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Graph 3: RAM vs Price

plt.figure(figsize=(9, 5))
sns.boxplot(x="Ram", y="Price_euros", data=df)
plt.title("RAM vs Laptop Price")
plt.xlabel("RAM")
plt.ylabel("Price in Euros")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Graph 4: Laptop Type vs Price

plt.figure(figsize=(10, 5))
sns.boxplot(x="TypeName", y="Price_euros", data=df)
plt.title("Laptop Type vs Price")
plt.xlabel("Laptop Type")
plt.ylabel("Price in Euros")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

data = df.copy()

# Remove unnecessary columns

data.drop(
    columns=["laptop_ID", "Product"],
    inplace=True
)

# Convert RAM to numeric
# Works with values such as 8GB and 8.0

data["Ram"] = pd.to_numeric(
    data["Ram"].astype(str).str.extract(r"([\d.]+)", expand=False),
    errors="coerce"
)

# Convert Weight to numeric
# Works with values such as 1.37kg and 1.37

data["Weight"] = pd.to_numeric(
    data["Weight"].astype(str).str.extract(r"([\d.]+)", expand=False),
    errors="coerce"
)

# Check missing values after conversion

print("\nMissing Values After Preprocessing:")
print(data.isnull().sum())

# Remove rows with missing values, if any

data.dropna(inplace=True)

print("\nPreprocessed Data:")
print(data.head())

print("\nPreprocessed Data Types:")
print(data.dtypes)


# ============================================================
# 5. SPLIT FEATURES AND TARGET
# ============================================================

X = data.drop("Price_euros", axis=1)

y = data["Price_euros"]

print("\nIndependent Variables:")
print(X.head())

print("\nTarget Variable:")
print(y.head())


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

num_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

cat_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================================
# 8. FEATURE SCALING
# ============================================================

scaler = MinMaxScaler()

X_train_num = scaler.fit_transform(
    X_train[num_cols]
)

X_test_num = scaler.transform(
    X_test[num_cols]
)


# ============================================================
# 9. CATEGORICAL ENCODING
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

X_train_cat = encoder.fit_transform(
    X_train[cat_cols]
).toarray()

X_test_cat = encoder.transform(
    X_test[cat_cols]
).toarray()


# ============================================================
# 10. COMBINE NUMERICAL AND CATEGORICAL DATA
# ============================================================

X_train_final = np.hstack(
    (X_train_num, X_train_cat)
)

X_test_final = np.hstack(
    (X_test_num, X_test_cat)
)

print("\nFinal Training Shape:", X_train_final.shape)
print("Final Testing Shape:", X_test_final.shape)


# ============================================================
# 11. MODEL BUILDING
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train_final,
    y_train
)

print("\nMODEL TRAINED SUCCESSFULLY!")


# ============================================================
# 12. MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test_final)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("MAE:", round(mae, 2))

print("MSE:", round(mse, 2))

print("RMSE:", round(rmse, 2))

print("R2 Score:", round(r2, 4))

print("R2 Score Percentage:", round(r2 * 100, 2), "%")


# ============================================================
# 14. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.xlabel("Actual Laptop Price")

plt.ylabel("Predicted Laptop Price")

plt.title("Actual vs Predicted Laptop Prices")

plt.tight_layout()
plt.show()


# ============================================================
# 15. SAVE MODEL ARTIFACTS
# ============================================================

artifacts = {

    "model": model,

    "scaler": scaler,

    "encoder": encoder,

    "num_cols": num_cols,

    "cat_cols": cat_cols

}

joblib.dump(
    artifacts,
    "laptop_price_artifacts.pkl"
)

print("\nMODEL ARTIFACTS SAVED SUCCESSFULLY!")

print("\nPROJECT TRAINING COMPLETED!")
