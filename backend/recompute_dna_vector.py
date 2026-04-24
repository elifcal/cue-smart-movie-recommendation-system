"""
recompute_dna_vector.py
"""

import os
import time
from dotenv import load_dotenv
from supabase import create_client
from ml.dna_scorer import dna_vector

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

TABLO = "film_dna"

def recompute(offset: int = 0, limit: int = None):
    filmler = []
    sayfa = 0
    sayfa_boyutu = 1000
    while True:
        sonuc = (
            supabase.table(TABLO)
            .select("tmdb_id, title, tempo, energy, speech_ratio, brightness, saturation, warmth, emotion_curve")
            .range(sayfa * sayfa_boyutu, (sayfa + 1) * sayfa_boyutu - 1)
            .execute().data
        )
        if not sonuc:
            break
        filmler.extend(sonuc)
        if len(sonuc) < sayfa_boyutu:
            break
        sayfa += 1

    filmler = filmler[offset:]
    if limit:
        filmler = filmler[:limit]

    print(f"Toplam işlenecek: {len(filmler)} film\n")
    guncelle = 0
    hata = 0

    for film in filmler:
        tmdb_id = film["tmdb_id"]
        try:
            ses_v = {
                "tempo":        film["tempo"],
                "energy":       film["energy"],
                "speech_ratio": film["speech_ratio"],
            }
            gorsel_v = {
                "brightness": film["brightness"],
                "saturation": film["saturation"],
                "warmth":     film["warmth"],
            }

            yeni_vektor = dna_vector(ses_v, gorsel_v, film["emotion_curve"])

            supabase.table(TABLO).update(
                {"dna_vector": yeni_vektor.tolist()}
            ).eq("tmdb_id", tmdb_id).execute()

            print(f"✅ [{tmdb_id}] {film.get('title', '?')}")
            guncelle += 1
        except Exception as e:
            print(f"❌ [{tmdb_id}] {e}")
            hata += 1

        time.sleep(0.1)

    print(f"\n✅ Güncellendi: {guncelle}  ❌ Hata: {hata}")

if __name__ == "__main__":
    recompute()
