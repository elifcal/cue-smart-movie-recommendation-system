"""
dna_storage.py
==============
DNA verisini Supabase'e kaydeder.
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from ml.dna_scorer import dna_vector
from precompute.subtitle_emotion import fused_emotion_curve

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_dna_to_supabase(
    tmdb_id: int,
    ses_verisi: Dict[str, Any],
    gorsel_verisi: Dict[str, Any],
    title: Optional[str] = None,
    whisper_text: Optional[str] = None,
    analyzed_at: Optional[str] = None
) -> Optional[Dict]:
    """
    Film DNA verisini kaydeder (UPSERT)
    """
    try:
        emotion_curve = fused_emotion_curve(tmdb_id)
        vektor = dna_vector(ses_verisi, gorsel_verisi, emotion_curve)

        if analyzed_at is None:
            analyzed_at = datetime.utcnow().isoformat()

        data = {
            "tmdb_id": tmdb_id,
            "title": title,
            "whisper_text": whisper_text,
            "tempo": ses_verisi.get("tempo"),
            "energy": ses_verisi.get("energy"),
            "speech_ratio": ses_verisi.get("speech_ratio"),
            "emotion_curve": emotion_curve,
            "brightness": gorsel_verisi.get("brightness"),
            "saturation": gorsel_verisi.get("saturation"),
            "warmth": gorsel_verisi.get("warmth"),
            "dna_vector": vektor.tolist(),
            "analyzed_at": analyzed_at,
        }

        data = {k: v for k, v in data.items() if v is not None}

        response = (
            supabase
            .table("film_dna")
            .upsert(data, on_conflict="tmdb_id")
            .execute()
        )

        if response.data:
            print(f"✅ Kaydedildi: {tmdb_id}")
            return response.data[0]
        print(f"⚠️ Response boş: {tmdb_id}")
        return None
    except Exception as e:
        print(f"❌ Supabase hata (ID: {tmdb_id}): {e}")
        return None
