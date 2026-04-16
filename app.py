from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import threading
import time
from datetime import datetime

app = Flask(__name__)

# LOAD MODEL
MODEL_PATH = "models/retrained_superstore_profit_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    model_loaded = True
else:
    model = None
    model_loaded = False
    print("Model file not found!")

# CATEGORY OPTIONS
SHIP_MODE = ["Standard Class", "Second Class", "First Class", "Same Day"]
SEGMENT = ["Consumer", "Corporate", "Home Office"]
REGION = ["East", "West", "Central", "South"]
CATEGORY = ["Furniture", "Office Supplies", "Technology"]
SUBCATEGORY = ["Chairs", "Tables", "Phones", "Storage"]

# ENCODING
ship_map = {name: i for i, name in enumerate(SHIP_MODE)}
segment_map = {name: i for i, name in enumerate(SEGMENT)}
region_map = {name: i for i, name in enumerate(REGION)}
category_map = {name: i for i, name in enumerate(CATEGORY)}
subcategory_map = {name: i for i, name in enumerate(SUBCATEGORY)}

# METRICS
app_metrics = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_predictions": 0,
    "last_prediction_time": None,
    "model_status": "loaded" if model_loaded else "not_loaded"
}

# ===============================
# ROOT (PROJECT STATUS)
# ===============================
@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "🚀 Superstore Profit Prediction API is running",
        "model_status": app_metrics["model_status"]
    })


# ===============================
# HEALTH CHECK
# ===============================
@app.route("/health", methods=["GET"])
def health():
    return jsonify(app_metrics)


# ===============================
# PREDICT API (JSON ONLY)
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Send JSON data only"}), 400

        data = request.get_json()

        # VALIDATION
        required_fields = [
            "ship_mode", "segment", "region", "category",
            "sub_category", "sales", "quantity",
            "discount", "order_month", "ship_duration"
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # ENCODING
        ship = ship_map[data["ship_mode"]]
        seg = segment_map[data["segment"]]
        reg = region_map[data["region"]]
        cat = category_map[data["category"]]
        sub = subcategory_map[data["sub_category"]]

        # CREATE DATAFRAME
        input_df = pd.DataFrame([{
            "Ship Mode": ship,
            "Segment": seg,
            "Region": reg,
            "Category": cat,
            "Sub-Category": sub,
            "Sales": float(data["sales"]),
            "Quantity": int(data["quantity"]),
            "Discount": float(data["discount"]),
            "order_month": int(data["order_month"]),
            "ship_duration": int(data["ship_duration"])
        }])

        # PREDICT
        prediction = model.predict(input_df)[0]
        prediction = round(float(prediction), 2)

        # UPDATE METRICS
        app_metrics["total_predictions"] += 1
        app_metrics["last_prediction_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            "status": "success",
            "predicted_profit": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ===============================
# BACKGROUND MONITORING (60 sec)
# ===============================
def monitor():
    while True:
        print("\n📊 MONITORING LOG")
        print("Time:", datetime.now())
        print("Total Predictions:", app_metrics["total_predictions"])
        print("Last Prediction:", app_metrics["last_prediction_time"])
        print("Model Status:", app_metrics["model_status"])
        print("-------------------------\n")
        time.sleep(60)

threading.Thread(target=monitor, daemon=True).start()


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)