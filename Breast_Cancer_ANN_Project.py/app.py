# ---------------------------------------------------------
# Imports & Memory Constraints (MUST BE AT THE TOP)
# ---------------------------------------------------------
import os
import logging

# Set TF logging to minimum and restrict threading to save RAM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, render_template, request, jsonify
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# App Configuration
# ---------------------------------------------------------
# Get absolute path for the folder Render is currently in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Correct pathing based on your folder structure
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_cancer_predictor.h5")
THRESHOLD = 0.5

# ---------------------------------------------------------
# Load Model & Preprocessing Objects
# ---------------------------------------------------------
logger.info(f"Attempting memory-optimized load from: {MODEL_PATH}")

# Global variable for the model
model = None

if os.path.exists(MODEL_PATH):
    try:
        # Using compile=False saves memory during the loading process
        model = load_model(MODEL_PATH, compile=False)
        logger.info("✅ Model loaded successfully (Memory Optimized)")
    except Exception as e:
        logger.error(f"❌ Load failed: {e}")
else:
    logger.error("❌ Model file not found!")

# Load dataset once to fit the scaler
data = load_breast_cancer(as_frame=True)
X = data.data
FEATURE_NAMES = list(X.columns)
scaler = StandardScaler()
scaler.fit(X)

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    prediction_class = None

    if request.method == "POST":
        try:
            if model is None:
                raise Exception("Model is not loaded on the server.")

            input_features = [float(request.form[feature]) for feature in FEATURE_NAMES]
            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # verbose=0 reduces logging memory overhead
            prob = float(model.predict(input_scaled, verbose=0)[0][0])
            probability = round(prob, 4)

            if prob >= THRESHOLD:
                result = "Benign Tumor"
                prediction_class = "benign"
            else:
                result = "Malignant Tumor"
                prediction_class = "malignant"

            logger.info(f"Prediction made: {result}")

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            result = "Error in input values or model availability."
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
# Application Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Dynamically find the port Render assigns
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)