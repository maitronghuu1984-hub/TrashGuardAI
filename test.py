import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "trashguard_mobilenetv2.h5"
IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Không đọc được ảnh")
        return

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    prediction = model.predict(image)[0][0]

    if prediction >= 0.5:
        label = "San truong co rac"
        confidence = prediction
    else:
        label = "San truong sach se"
        confidence = 1 - prediction

    print("Kết quả:", label)
    print("Độ tin cậy:", round(float(confidence), 2))


predict_image("test3.jpg")