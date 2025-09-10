# key.py: Garbage Classification API with FastAPI

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import uvicorn

# --- Configuration & Initialization ---
load_dotenv()

MODEL_PATH = 'garbage_classification_model.h5'

# Class names (order must match training)
class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes',
    'trash', 'white-glass'
]

# --- Load Model ---
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        dummy_input = np.zeros((1, 224, 224, 3))
        model.predict(dummy_input, verbose=0)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

model = load_model()

# --- FastAPI App ---
app = FastAPI(
    title="Garbage Classification API",
    description="A simple API to classify garbage items."
)

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_index])

        prob_dict = {class_name: float(predictions[0][i]) for i, class_name in enumerate(class_names)}

        return JSONResponse(content={
            "prediction": class_names[predicted_class_index],
            "confidence": confidence,
            "probabilities": prob_dict
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


# --- Autorun Server (no reload here) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
