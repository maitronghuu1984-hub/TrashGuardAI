from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import time
import uvicorn

app = FastAPI(
    title="TrashGuard AI Server",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "trashguard_mobilenetv2.h5"
IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

last_warning_time = 0
WARNING_COOLDOWN = 20

CONFIDENCE_THRESHOLD = 0.80


def predict_trash(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    prediction = model.predict(image)[0][0]

    if prediction >= 0.5:
        label = "trash"
        confidence = float(prediction)
    else:
        label = "clean"
        confidence = float(1 - prediction)

    return label, confidence


@app.get("/")
def home():
    return {
        "project": "TrashGuard AI",
        "model": "MobileNetV2",
        "status": "Server is running"
    }


@app.post("/analyze")
async def analyze(request: Request):
    global last_warning_time

    try:
        image_bytes = await request.body()

        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={
                    "trash_detected": False,
                    "label": "none",
                    "confidence": 0.0,
                    "action": "NONE",
                    "message": "No image received"
                }
            )

        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "trash_detected": False,
                    "label": "none",
                    "confidence": 0.0,
                    "action": "NONE",
                    "message": "Cannot decode image"
                }
            )

        label, confidence = predict_trash(image)

        current_time = time.time()

        if label == "trash" and confidence >= CONFIDENCE_THRESHOLD:
            if current_time - last_warning_time >= WARNING_COOLDOWN:
                last_warning_time = current_time

                return {
                    "trash_detected": True,
                    "label": label,
                    "confidence": round(confidence, 2),
                    "action": "PLAY_WARNING",
                    "message": "Trash detected on school yard"
                }

            return {
                "trash_detected": True,
                "label": label,
                "confidence": round(confidence, 2),
                "action": "WAIT",
                "message": "Trash detected but warning cooldown active"
            }

        return {
            "trash_detected": False,
            "label": label,
            "confidence": round(confidence, 2),
            "action": "NONE",
            "message": "School yard is clean"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "trash_detected": False,
                "label": "error",
                "confidence": 0.0,
                "action": "NONE",
                "message": str(e)
            }
        )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )