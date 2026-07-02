from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import tensorflow as tf
import numpy as np
import cv2
import time
import uvicorn
import os
import json

app = FastAPI(title="TrashGuard AI Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "trashguard_mobilenetv2.h5"
IMG_SIZE = 224
UPLOAD_DIR = "uploads"
HISTORY_FILE = "history.jsonl"

os.makedirs(UPLOAD_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

last_warning_time = 0
WARNING_COOLDOWN = 20
CONFIDENCE_THRESHOLD = 0.80
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def now_vn():
    return datetime.now(VN_TZ)


def predict_trash(image):
    image_input = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    image_input = np.expand_dims(image_input, axis=0)
    image_input = tf.keras.applications.mobilenet_v2.preprocess_input(image_input)

    prediction = model.predict(image_input, verbose=0)[0][0]

    if prediction >= 0.5:
        return "trash", float(prediction)
    return "clean", float(1 - prediction)


def draw_result_on_image(image, label, confidence, action):
    annotated = image.copy()

    color = (0, 0, 255) if label == "trash" else (0, 180, 0)
    text = f"TrashGuard AI: {label.upper()} | Confidence: {confidence:.2f} | Action: {action}"

    cv2.rectangle(annotated, (10, 10), (annotated.shape[1] - 10, 70), color, -1)
    cv2.putText(
        annotated,
        text,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return annotated


def save_history(record):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_history(limit=100):
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except:
            pass

    return records[::-1]


def get_today_count():
    today = now_vn().strftime("%Y-%m-%d")
    records = read_history(limit=10000)

    return sum(
        1 for r in records
        if r.get("date") == today and r.get("trash_detected") is True
    )


def get_last_7_days_stats():
    records = read_history(limit=10000)
    result = []

    for i in range(6, -1, -1):
        day = now_vn() - timedelta(days=i)
        date_str = day.strftime("%Y-%m-%d")

        count = sum(
            1 for r in records
            if r.get("date") == date_str and r.get("trash_detected") is True
        )

        result.append({
            "date": date_str,
            "count": count
        })

    return result


@app.get("/")
def home():
    return {
        "project": "TrashGuard AI",
        "model": "MobileNetV2",
        "version": "2.0.0",
        "status": "Server is running",
        "dashboard": "/dashboard",
        "last_image": "/last-image",
        "history": "/history"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "TrashGuard AI API is healthy"
    }


@app.post("/analyze")
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    global last_warning_time

    try:
        if file is not None:
            image_bytes = await file.read()
        else:
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

        timestamp = now_vn()
        time_code = timestamp.strftime("%Y%m%d_%H%M%S")

        raw_path = os.path.join(UPLOAD_DIR, f"{time_code}_raw.jpg")
        with open(raw_path, "wb") as f:
            f.write(image_bytes)

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
        action = "NONE"
        trash_detected = False
        message = "School yard is clean"

        if label == "trash" and confidence >= CONFIDENCE_THRESHOLD:
            trash_detected = True

            if current_time - last_warning_time >= WARNING_COOLDOWN:
                last_warning_time = current_time
                action = "PLAY_WARNING"
                message = "Trash detected on school yard"
            else:
                action = "WAIT"
                message = "Trash detected but warning cooldown active"

        annotated = draw_result_on_image(image, label, confidence, action)

        annotated_path = os.path.join(UPLOAD_DIR, f"{time_code}_annotated.jpg")
        latest_path = os.path.join(UPLOAD_DIR, "latest.jpg")

        cv2.imwrite(annotated_path, annotated)
        cv2.imwrite(latest_path, annotated)

        record = {
            "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "date": timestamp.strftime("%Y-%m-%d"),
            "label": label,
            "confidence": round(confidence, 2),
            "trash_detected": trash_detected,
            "action": action,
            "raw_image": raw_path,
            "annotated_image": annotated_path,
            "message": message
        }

        save_history(record)

        return {
            "trash_detected": trash_detected,
            "label": label,
            "confidence": round(confidence, 2),
            "action": action,
            "message": message,
            "last_image_url": "/last-image",
            "dashboard_url": "/dashboard"
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


@app.get("/last-image")
def last_image():
    latest_path = os.path.join(UPLOAD_DIR, "latest.jpg")

    if not os.path.exists(latest_path):
        return JSONResponse(
            status_code=404,
            content={"message": "No image uploaded yet"}
        )

    return FileResponse(latest_path, media_type="image/jpeg")


@app.get("/history")
def history():
    return {
        "total_records": len(read_history(limit=10000)),
        "records": read_history(limit=100)
    }


@app.get("/stats")
def stats():
    return {
        "today_trash_count": get_today_count(),
        "last_7_days": get_last_7_days_stats()
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    records = read_history(limit=20)
    today_count = get_today_count()
    stats_7_days = get_last_7_days_stats()

    max_count = max([d["count"] for d in stats_7_days] + [1])

    chart_html = ""
    for d in stats_7_days:
        width = int((d["count"] / max_count) * 100) if max_count > 0 else 0
        chart_html += f"""
        <div class="bar-row">
            <div class="date">{d["date"]}</div>
            <div class="bar-bg">
                <div class="bar" style="width:{width}%"></div>
            </div>
            <div class="count">{d["count"]}</div>
        </div>
        """

    history_rows = ""
    for r in records:
        badge_class = "trash" if r.get("trash_detected") else "clean"
        history_rows += f"""
        <tr>
            <td>{r.get("time")}</td>
            <td><span class="badge {badge_class}">{r.get("label")}</span></td>
            <td>{r.get("confidence")}</td>
            <td>{r.get("action")}</td>
            <td>{r.get("message")}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="10">
        <title>TrashGuard AI Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                margin: 0;
                padding: 20px;
                color: #1f2937;
            }}
            .container {{
                max-width: 1200px;
                margin: auto;
            }}
            h1 {{
                color: #0f766e;
                margin-bottom: 5px;
            }}
            .subtitle {{
                color: #555;
                margin-bottom: 25px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            .card {{
                background: white;
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }}
            .image-box img {{
                width: 100%;
                border-radius: 12px;
                border: 1px solid #ddd;
            }}
            .big-number {{
                font-size: 56px;
                color: #dc2626;
                font-weight: bold;
            }}
            .bar-row {{
                display: grid;
                grid-template-columns: 120px 1fr 40px;
                align-items: center;
                gap: 10px;
                margin: 10px 0;
            }}
            .bar-bg {{
                background: #e5e7eb;
                border-radius: 20px;
                height: 18px;
                overflow: hidden;
            }}
            .bar {{
                background: #0f766e;
                height: 100%;
                border-radius: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            th, td {{
                padding: 10px;
                border-bottom: 1px solid #e5e7eb;
                text-align: left;
            }}
            th {{
                background: #f0fdfa;
                color: #0f766e;
            }}
            .badge {{
                padding: 4px 10px;
                border-radius: 12px;
                color: white;
                font-weight: bold;
            }}
            .trash {{
                background: #dc2626;
            }}
            .clean {{
                background: #16a34a;
            }}
            .footer {{
                margin-top: 20px;
                color: #777;
                font-size: 13px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TrashGuard AI Dashboard</h1>
            <div class="subtitle">
                Hệ thống AI phát hiện rác trên sân trường và phát loa nhắc nhở thu gom rác đúng nơi quy định
            </div>

            <div class="grid">
                <div class="card image-box">
                    <h2>Ảnh mới nhất</h2>
                    <img src="/last-image?time={int(time.time())}" alt="Latest image">
                </div>

                <div class="card">
                    <h2>Số lần phát hiện rác hôm nay</h2>
                    <div class="big-number">{today_count}</div>
                    <p>Thống kê theo ngày hiện tại tại Việt Nam.</p>

                    <h3>Biểu đồ 7 ngày gần nhất</h3>
                    {chart_html}
                </div>
            </div>

            <div class="card" style="margin-top:20px;">
                <h2>Nhật ký cảnh báo gần đây</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Thời gian</th>
                            <th>Nhãn</th>
                            <th>Độ tin cậy</th>
                            <th>Hành động</th>
                            <th>Thông báo</th>
                        </tr>
                    </thead>
                    <tbody>
                        {history_rows}
                    </tbody>
                </table>
            </div>

            <div class="footer">
                Trang tự động làm mới sau mỗi 10 giây. Server version 2.0.0.
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )