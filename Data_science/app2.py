import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Dev CORS helper
from flask_cors import CORS

# ------------------------------
# Configuration
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cleaned_police_overtime_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.json") 

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bpd_backend")

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__, static_folder="assets", template_folder=".")

# ------------------------------
# CORS (development)
# ------------------------------
# Allow only localhost dev origins by default. If you serve the frontend via
# python -m http.server:8000, include http://localhost:8000 below.
CORS(
    app,
    resources={
        r"/predict": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:8000"]},
        r"/allowed_titles": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:8000"]},
    },
    supports_credentials=False,
    
)

# ------------------------------
# Load dataset and encoder
# ------------------------------
if not os.path.exists(CSV_PATH):
    logger.error("CSV file not found at %s", CSV_PATH)
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
logger.info("Loaded CSV with %d rows and %d columns", df.shape[0], df.shape[1])

# Fit a LabelEncoder on TITLE column (used for incoming requests)
if "TITLE" not in df.columns:
    logger.error("TITLE column not present in CSV")
    raise KeyError("TITLE column not present in CSV")

le = LabelEncoder()
# fill NA with placeholder to avoid issues
titles = df["TITLE"].fillna("UNKNOWN_TITLE").astype(str).values
le.fit(titles)
ALLOWED_TITLES = list(le.classes_)

# ------------------------------
# Load XGBoost model
# ------------------------------
if not os.path.exists(MODEL_PATH):
    logger.error("Model file not found at %s", MODEL_PATH)
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

xgb_model = XGBRegressor()
xgb_model.load_model(MODEL_PATH)
logger.info("Loaded XGBoost model from %s", MODEL_PATH)

# ------------------------------
# Feature definitions (order matters)
# ------------------------------
FEATURES = ["REGULAR", "RETRO", "OTHER", "INJURED", "DETAIL", "POSTAL", "TITLE_ENC"]

# ------------------------------
# Routes
# ------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")



@app.route("/model")
def model_page():
    return render_template("model_try.html")


@app.route("/allowed_titles", methods=["GET"])
def allowed_titles():
    return jsonify({"allowed_titles": ALLOWED_TITLES})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept either JSON or form-encoded POST
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()

        required = ["REGULAR", "RETRO", "OTHER", "INJURED", "DETAIL", "POSTAL", "TITLE"]
        missing = [k for k in required if k not in data or str(data.get(k)).strip() == ""]
        if missing:
            return jsonify({"error": "missing_fields", "message": f"Missing fields: {', '.join(missing)}"}), 400

        # Parse numeric fields with safe conversion
        try:
            REGULAR = float(data["REGULAR"])
            RETRO = float(data["RETRO"])
            OTHER = float(data["OTHER"])
            INJURED = float(data["INJURED"])
            DETAIL = float(data["DETAIL"])
            POSTAL = float(data["POSTAL"])
        except ValueError as e:
            return jsonify({"error": "bad_input", "message": "Numeric fields must be numbers."}), 400

        TITLE = str(data["TITLE"]).strip()
        if TITLE == "":
            return jsonify({"error": "bad_input", "message": "TITLE cannot be empty."}), 400

        # Handle TITLE encoding - return helpful error if unknown
        if TITLE not in ALLOWED_TITLES:
            # Suggest top 10 similar titles (simple substring match) to be user-friendly
            suggestions = [t for t in ALLOWED_TITLES if TITLE.lower() in t.lower()]
            return jsonify({
                "error": "unknown_title",
                "message": f"Unknown TITLE '{TITLE}'.",
                "suggestions": suggestions[:10],
                "allowed_count": len(ALLOWED_TITLES)
            }), 400

        TITLE_ENC = int(le.transform([TITLE])[0])

        # Build input dataframe in exact feature order
        input_row = {
            "REGULAR": REGULAR,
            "RETRO": RETRO,
            "OTHER": OTHER,
            "INJURED": INJURED,
            "DETAIL": DETAIL,
            "POSTAL": POSTAL,
            "TITLE_ENC": TITLE_ENC
        }
        input_df = pd.DataFrame([input_row], columns=FEATURES)

        # Optional: ensure dtypes
        input_df = input_df.astype(float)

        # Predict using the loaded XGBoost model (variable name is xgb_model)
        pred = xgb_model.predict(input_df)
        result = float(pred[0])

        # Return prediction (rounded for readability)
        return jsonify({"prediction": round(result, 2)}), 200

    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify({"error": "server_error", "message": "Error during prediction", "detail": str(exc)}), 500


# Static file helper (if needed to serve JS/CSS from assets/)
@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(os.path.join(BASE_DIR, "assets"), filename)


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
