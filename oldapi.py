import os, json, logging, base64
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, Model

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEQ_LEN  = 30
FEAT     = 258
D_MODEL  = 256
HEADS    = 8
FF       = 512
LSTM     = 256
DROPOUT  = 0.3
TOP_K    = 5
NUM_POSE = 33 * 4
NUM_HAND = 21 * 3

mp_holistic = mp.solutions.holistic
_model      = None
_idx2label  = None

class TransformerBlock(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mha   = layers.MultiHeadAttention(num_heads=HEADS, key_dim=D_MODEL//HEADS, dropout=0.1)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(FF, activation="gelu"),
            layers.Dropout(0.1),
            layers.Dense(D_MODEL)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(0.1)
        self.drop2 = layers.Dropout(0.1)

    def call(self, x, training=False):
        h = self.norm1(x)
        h = self.mha(h, h, training=training)
        h = self.drop1(h, training=training)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h, training=training)
        h = self.drop2(h, training=training)
        return x + h

    def get_config(self):
        return super().get_config()

def extract_keypoints(results) -> np.ndarray:
    pose = np.array([[r.x,r.y,r.z,r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(NUM_POSE)
    lh   = np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(NUM_HAND)
    rh   = np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(NUM_HAND)
    return np.concatenate([pose, lh, rh])

def normalize_seq(frames_kp: list) -> np.ndarray:
    seq = np.array(frames_kp, dtype=np.float32)
    T   = seq.shape[0]
    if T >= SEQ_LEN:
        start = (T - SEQ_LEN) // 2
        seq   = seq[start: start + SEQ_LEN]
    else:
        seq = np.vstack([seq, np.zeros((SEQ_LEN - T, FEAT), dtype=np.float32)])
    mean = seq.mean()
    std  = seq.std() + 1e-6
    seq  = (seq - mean) / std
    return seq[np.newaxis, ...]

def frames_to_seq(frames_b64: List[str]) -> np.ndarray:
    frames_kp = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
        for b64 in frames_b64:
            if "," in b64: b64 = b64.split(",", 1)[1]
            try:
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                frames_kp.append(extract_keypoints(h.process(image)))
            except: continue
    if not frames_kp:
        raise ValueError("No valid frames")
    return normalize_seq(frames_kp)

def run_inference(seq: np.ndarray) -> list:
    probs       = _model.predict(seq, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:TOP_K]
    return [
        {"rank": i+1, "label": _idx2label[str(idx)], "confidence": round(float(probs[idx])*100, 2)}
        for i, idx in enumerate(top_indices)
    ]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _idx2label
    model_dir  = os.environ.get("MODEL_DIR", r"D:\Project\model")
    with open(os.path.join(model_dir, "label_map.json"), encoding="utf-8") as f:
        _idx2label = json.load(f)
    _model = tf.keras.models.load_model(
        os.path.join(model_dir, "best.keras"),
        custom_objects={"TransformerBlock": TransformerBlock}
    )
    logger.info(f"✅ Model loaded — {len(_idx2label)} classes")
    yield

app = FastAPI(title="Sign Language API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class FramesRequest(BaseModel):
    frames: List[str] = Field(..., min_length=5, max_length=300)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "num_classes": len(_idx2label) if _idx2label else 0}

@app.post("/predict")
async def predict(body: FramesRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        preds = run_inference(frames_to_seq(body.frames))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {
        "best_label": preds[0]["label"],
        "best_confidence": preds[0]["confidence"],
        "top_predictions": preds
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
