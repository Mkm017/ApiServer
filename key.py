# key.py: The Backend API with FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import mysql.connector

# --- Configuration & Initialization ---
# Load environment variables from .env file
load_dotenv()

# Define paths
MODEL_PATH = 'garbage_classification_model.h5'
DATA_DIR = 'Data'

# Get class names from the data directory
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"The directory '{DATA_DIR}' does not exist.")
class_names = sorted(os.listdir(DATA_DIR))

# --- Database Connection ---
def get_db_connection():
    """Establishes and returns a new MySQL database connection."""
    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

# --- Model Loading ---
def load_model():
    """Loads the pre-trained TensorFlow model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # For a pre-trained model, a dummy run can initialize it on startup
        dummy_input = np.zeros((1, 224, 224, 3))
        model.predict(dummy_input, verbose=0)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

# Load the model once at startup
model = load_model()

# Initialize the FastAPI app
app = FastAPI(
    title="Garbage Classification API",
    description="A simple API to classify garbage items."
)

# --- Database Logger Function ---
def log_api_call(db_conn, client_ip: str, endpoint: str):
    """Inserts a new record into the api_calls table."""
    try:
        cursor = db_conn.cursor()
        query = "INSERT INTO api_calls (client_ip, endpoint) VALUES (%s, %s)"
        cursor.execute(query, (client_ip, endpoint))
        db_conn.commit()
    except mysql.connector.Error as e:
        print(f"Error logging API call: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()

# --- API Endpoint ---
@app.post("/classify/")
async def classify_image(request: Request, file: UploadFile = File(...)):
    """Receives an image, processes it, and returns the classification prediction."""
    client_ip = request.client.host
    db_conn = get_db_connection()
    if db_conn:
        log_api_call(db_conn, client_ip, "/classify/")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Process the image
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        
        # Make a prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get results
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_index])
        
        prob_dict = {class_name: float(predictions[0][i]) for i, class_name in enumerate(class_names)}
        
        response = {
            "prediction": class_names[predicted_class_index],
            "confidence": confidence,
            "probabilities": prob_dict
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        if db_conn:
            db_conn.close()
