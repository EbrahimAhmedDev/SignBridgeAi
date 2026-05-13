import os, json, logging, base64
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers

from google import genai
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ─── Model Constants ──────────────────────────────────────────────────────────
SEQ_LEN  = 30
FEAT     = 258
D_MODEL  = 256
HEADS    = 8
FF       = 512
DROPOUT  = 0.3
TOP_K    = 5
NUM_POSE = 33 * 4
NUM_HAND = 21 * 3

mp_holistic = mp.solutions.holistic
_model      = None
_idx2label  = None

# ─── Config from environment ──────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", ".")
AI_HOST = os.environ.get("AI_HOST", "127.0.0.1")
AI_PORT = int(os.environ.get("AI_PORT", "8000"))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ─── Transformer Block ────────────────────────────────────────────────────────
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

# ─── Keypoints Extraction ─────────────────────────────────────────────────────
def extract_keypoints(results) -> np.ndarray:
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(NUM_POSE)
    lh   = np.array([[r.x, r.y, r.z]
                     for r in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(NUM_HAND)
    rh   = np.array([[r.x, r.y, r.z]
                     for r in results.right_hand_landmarks.landmark]).flatten() \
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
    return ((seq - mean) / std)[np.newaxis, ...]

def frames_to_seq(frames_b64: List[str]) -> np.ndarray:
    """Convert base64 frames list to one sequence (single word prediction)"""
    frames_kp = []
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as h:
        for b64 in frames_b64:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                frame = cv2.imdecode(
                    np.frombuffer(base64.b64decode(b64), np.uint8),
                    cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                frames_kp.append(extract_keypoints(h.process(image)))
            except:
                continue
    if not frames_kp:
        raise ValueError("No valid frames")
    return normalize_seq(frames_kp)

def run_inference(seq: np.ndarray) -> list:
    probs       = _model.predict(seq, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:TOP_K]
    return [
        {
            "rank": i + 1,
            "label": _idx2label[str(idx)],
            "confidence": round(float(probs[idx]) * 100, 2)
        }
        for i, idx in enumerate(top_indices)
    ]

# ─── Sliding Window: long video → list of words ───────────────────────────────
def video_to_words(all_frames_b64: List[str],
                   step: int = 5,
                   min_confidence: float = 45.0) -> List[str]:
    """
    Takes all sentence frames and extracts words using sliding window.
    Each window = 30 frames, moves every (step) frames.
    Removes consecutive duplicates.
    """
    all_kp = []

    # Extract keypoints once for all frames
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as h:
        for b64 in all_frames_b64:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                frame = cv2.imdecode(
                    np.frombuffer(base64.b64decode(b64), np.uint8),
                    cv2.IMREAD_COLOR
                )
                if frame is None:
                    all_kp.append(np.zeros(FEAT))
                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                all_kp.append(extract_keypoints(h.process(image)))
            except:
                all_kp.append(np.zeros(FEAT))

    total = len(all_kp)
    words = []

    # Sliding window
    for start in range(0, total - SEQ_LEN + 1, step):
        window_kp = all_kp[start: start + SEQ_LEN]
        seq       = normalize_seq(window_kp)
        preds     = run_inference(seq)
        best      = preds[0]

        if best["confidence"] >= min_confidence:
            words.append(best["label"])

    # Remove consecutive duplicates
    deduped = []
    for w in words:
        if not deduped or deduped[-1] != w:
            deduped.append(w)

    return deduped

# ─── Gemini: words → sentence ─────────────────────────────────────────────────
def words_to_sentence(words: List[str]) -> str:
    """Sends detected words to Gemini to form a natural English sentence"""
    if not words:
        return ""

    words_str = " - ".join(words)

    prompt = f"""You are a sign language translation assistant.
Recognized words in order: {words_str}
Form a natural grammatically correct English sentence from these words.
Return ONLY the sentence, no explanation."""

    if client is None:
        logger.warning("GEMINI_API_KEY is not set; returning joined words.")
        return " ".join(words)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        # If Gemini fails, return words joined
        return " ".join(words)

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _idx2label
    with open(os.path.join(MODEL_DIR, "label_map.json"), encoding="utf-8") as f:
        _idx2label = json.load(f)
    _model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "best.keras"),
        custom_objects={"TransformerBlock": TransformerBlock}
    )
    logger.info(f"Model loaded — {len(_idx2label)} classes")
    yield

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Sign Language API", version="4.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─── Request Models ───────────────────────────────────────────────────────────
class FramesRequest(BaseModel):
    frames: List[str] = Field(..., min_length=5, max_length=300)

class SentenceRequest(BaseModel):
    frames: List[str] = Field(
        ..., min_length=5, max_length=3000,
        description="All frames of the full sentence"
    )
    step: int = Field(
        default=5, ge=5, le=30,
        description="Sliding window step"
    )
    min_confidence: float = Field(
        default=45.0, ge=0.0, le=100.0,
        description="Minimum confidence to accept a word"
    )

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "num_classes": len(_idx2label) if _idx2label else 0
    }

@app.post("/predict")
async def predict(body: FramesRequest):
    """Predict a single word — original endpoint"""
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        preds = run_inference(frames_to_seq(body.frames))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {
        "best_label":      preds[0]["label"],
        "best_confidence": preds[0]["confidence"],
        "top_predictions": preds
    }

@app.post("/predict_sentence")
async def predict_sentence(body: SentenceRequest):
    """
    Translate a full sentence from sign language.
    Many frames → sliding window → words → Gemini → sentence
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        # 1. Extract words
        words = video_to_words(
            body.frames,
            step=body.step,
            min_confidence=body.min_confidence
        )

        if not words:
            raise HTTPException(
                422,
                "No words detected. Try recording more clearly or for longer."
            )

        # 2. Build sentence with Gemini
        sentence = words_to_sentence(words)

        return {
            "sentence":       sentence,
            "words_detected": words,
            "words_count":    len(words)
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"predict_sentence error: {e}")
        raise HTTPException(500, f"Internal error: {str(e)}")

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=AI_HOST, port=AI_PORT)
