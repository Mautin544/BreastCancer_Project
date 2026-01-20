# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from flask import Flask, render_template, request, jsonify
import numpy as np
import logging
import os

from tensorflow.keras.models import load_model
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# App Configuration
# ---------------------------------------------------------
# Explicitly setting template and static folders helps Render find them
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(BASE_DIR, "model", "model_cancer_predictor.h5")
THRESHOLD = 0.5
model = None # Initialize as None to prevent crashes

# ---------------------------------------------------------
# Load Model & Preprocessing Objects
# ---------------------------------------------------------
logger.info(f"Checking model at: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        logger.info("✅ ANN Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
else:
    logger.error(f"❌ Model file NOT found at {MODEL_PATH}")

# Pre-fit the scaler (consistent with training)
try:
    data = load_breast_cancer(as_frame=True)
    X = data.data
    FEATURE_NAMES = list(X.columns)
    scaler = StandardScaler()
    scaler.fit(X)
    logger.info("✅ Scaler initialized and fitted")
except Exception as e:
    logger.error(f"❌ Failed to initialize scaler: {e}")

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    prediction_class = None

    if request.method == "POST":
        # Check if model exists before predicting
        if model is None:
            return render_template("index.html", features=FEATURE_NAMES, prediction="Model not available.", prediction_class="error")

        try:
            input_features = [float(request.form[feature]) for feature in FEATURE_NAMES]
            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Prediction logic
            prob = float(model.predict(input_scaled, verbose=0)[0][0])
            probability = round(prob, 4)

            if prob >= THRESHOLD:
                result = "Benign Tumor"
                prediction_class = "benign"
            else:
                result = "Malignant Tumor"
                prediction_class = "malignant"

            logger.info(f"Prediction: {result} ({probability})")

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            result = "Error in input values."
            prediction_class = "error"

    return render_template(
        "index.html",
        features=FEATURE_NAMES,
        prediction=result,
        prediction_class=prediction_class,
        probability=probability
    )

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(status="ok", model_loaded=(model is not None)), 200

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False) # Keep debug False for production