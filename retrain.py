import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor



# CONFIGURATION

DATA_PATH = "data/new_data.csv"
OLD_MODEL_PATH = "models/superstore_profit_model.pkl"
NEW_MODEL_PATH = "models/retrained_superstore_profit_model.pkl"



# LOAD NEW DATASET

df = pd.read_csv("new_data.csv")

print("New dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())


# CHECK REQUIRED COLUMNS

required_columns = [
    "Ship Mode", "Segment", "Region", "Category", "Sub-Category",
    "Sales", "Quantity", "Discount", "order_month", "ship_duration", "Profit"
]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in new dataset: {missing_cols}")



# ENCODE CATEGORICAL COLUMNS

categorical_cols = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le



# SPLIT FEATURES AND TARGET

X = df.drop("Profit", axis=1)
y = df["Profit"]



# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# LOAD OLD MODEL (IF AVAILABLE) AND RETRAIN

if os.path.exists(OLD_MODEL_PATH):
    old_model = joblib.load(OLD_MODEL_PATH)
    print(f"\nOld model loaded from: {OLD_MODEL_PATH}")
    print("Loaded model type:", type(old_model).__name__)

    # Refit the loaded model on the new dataset.
    # This works when the saved model supports fit() with the new feature set.
    try:
        model = old_model
        model.fit(X_train, y_train)
        print("Loaded model retrained successfully on the new dataset.")
    except Exception as e:
        print("\nLoaded model could not be retrained directly.")
        print("Reason:", e)
        print("Creating a fresh RandomForestRegressor using the new dataset instead...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
else:
    print(f"\nOld model not found at: {OLD_MODEL_PATH}")
    print("Training a fresh RandomForestRegressor on the new dataset...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)



# EVALUATE MODEL

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation")
print("R2 Score :", r2)
print("RMSE     :", rmse)


# =========================================================
# SAVE RETRAINED MODEL
# =========================================================
os.makedirs("models", exist_ok=True)
joblib.dump(model, NEW_MODEL_PATH)

print(f"\nRetrained model saved to: {NEW_MODEL_PATH}")
