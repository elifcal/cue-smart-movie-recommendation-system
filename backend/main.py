from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Cue API", version="0.1.0")

# CORS ayarı — Mehmet'in frontend'i buraya bağlanacak
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sağlık kontrolü
@app.get("/health")
async def health():
    return {"status": "ok", "project": "cue"}

# Arama endpoint'i — şimdilik sahte veri döndürüyor
@app.get("/search")
async def search(q: str):
    return {
        "query": q,
        "results": [
            {
                "id": 275,
                "title": "The Sixth Sense",
                "year": 1999,
                "vote_average": 8.1,
                "poster_path": "/fVPDEjs6TqDNMnqJaGrPKEJFMDM.jpg",
                "overview": "Bir çocuk psikologu, ölüleri görebilen bir çocukla çalışmaya başlar.",
                "why_text": "Korku türünde, 90'larda geçen, twist sonu olan bir film.",
                "emotion_curve": [0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.21, 0.19, 0.14, 0.08],
                "color_palette": ["#1a1a2e", "#16213e", "#0f3460"]
            }
        ]
    }

# Geri bildirim endpoint'i
@app.post("/feedback")
async def feedback(film_id: int, action: str):
    return {"status": "ok", "film_id": film_id, "action": action}