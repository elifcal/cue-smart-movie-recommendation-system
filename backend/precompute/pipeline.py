import os
import time
import requests
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# --- KONFİGÜRASYON ---
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Hedef sayılar (İhtiyaca göre güncellenebilir)
TARGETS = {
    "Dram": 550,
    "Aksiyon": 450,
    "Komedi": 400,
    "Gerilim": 380,
    "Korku": 300,
    "Suç": 320,
    "Bilim-Kurgu": 280,
    "Romantik": 280,
    "Macera": 250,
    "Animasyon": 200,
    "Tarih": 200,
    "Fantastik": 180,
    "Aile": 150,
    "Belgesel": 120,
    "Savaş": 120,
    "Müzik": 80,
    "Türkçe": 700
}

# --- YARDIMCI FONKSİYONLAR ---

def get_movie_details(movie_id, lang):
    """Belirli bir dilde film detaylarını çeker."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": lang,
        "append_to_response": "keywords,credits,videos"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        print(f"⚠️ Hata (ID: {movie_id}, Dil: {lang}): {e}")
        return None

def save_to_supabase(movie_id, original_lang):
    """
    Mantık:
    - Türkçe filmler için her şey Türkçe.
    - Diğer diller için her şey İngilizce.
    - Başlıklar: title(TR), english_title(EN), original_title(Orijinal)
    """
    # Verileri hem TR hem EN olarak çekiyoruz
    en_data = get_movie_details(movie_id, "en-US")
    tr_data = get_movie_details(movie_id, "tr-TR")

    if not en_data or not tr_data:
        return False

    # İçerik seçimi: Orijinal dil Türkçe ise TR veriyi, değilse EN veriyi baz al
    if original_lang == 'tr':
        main_data = tr_data
    else:
        main_data = en_data

    # Credits Filtreleme (Yönetmen + İlk 3 Oyuncu)
    all_credits = main_data.get("credits", {})
    cast = all_credits.get("cast", [])[:3]
    directors = [m for m in all_credits.get("crew", []) if m.get("job") == "Director"]

    filtered_credits = {
        "cast": [{"name": c.get("name"), "character": c.get("character")} for c in cast],
        "directors": [{"name": d.get("name")} for d in directors]
    }

    try:
        entry = {
            "tmdb_id": movie_id,
            "imdb_id": main_data.get("imdb_id"),

            # Üçlü Başlık Yapısı
            "title": tr_data.get("title"),              # Her zaman Türkçe ismi
            "english_title": en_data.get("title"),       # Her zaman İngilizce ismi
            "original_title": main_data.get("original_title"), # Orijinal ismi

            # İçerik (TR ise Türkçe, Yabancı ise İngilizce)
            "overview": main_data.get("overview"),
            "tagline": main_data.get("tagline"),
            "genres": main_data.get("genres"),
            "genre_ids": [g['id'] for g in main_data.get("genres", [])],
            "keywords": main_data.get("keywords", {}).get("keywords", []),

            # Diğer Veriler
            "vote_average": main_data.get("vote_average"),
            "vote_count": main_data.get("vote_count"),
            "popularity": main_data.get("popularity"),
            "poster_path": main_data.get("poster_path"),
            "release_date": main_data.get("release_date") if main_data.get("release_date") else None,
            "runtime": main_data.get("runtime"),
            "budget": main_data.get("budget"),
            "revenue": main_data.get("revenue"),
            "original_language": original_lang, # Buradaki typo düzeltildi
            "credits": filtered_credits,
            "videos": main_data.get("videos", {}).get("results", []),
            "production_countries": main_data.get("production_countries")
        }

        # Supabase'e gönder
        supabase.table("movies").upsert(entry).execute()
        return True
    except Exception as e:
        print(f"❌ Kayıt Hatası (ID: {movie_id}): {e}")
        return False

# --- ANA DÖNGÜ ---

def run_pipeline():
    # Tür listesini çek (Filtreleme için ID'ler lazım)
    res = requests.get(f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=tr-TR").json()
    genres_dict = {g['name'].lower(): g['id'] for g in res.get('genres', [])}

    for label, goal in TARGETS.items():
        count, page = 0, 1
        vote_limit = 10 if label == "Türkçe" else 100

        print(f"\n--- 🚀 {label} için {goal} film çekiliyor ---")

        while count < goal:
            params = {
                "api_key": TMDB_API_KEY,
                "sort_by": "popularity.desc",
                "page": page,
                "vote_count.gte": vote_limit,
                "language": "tr-TR" # Keşif için TR dili
            }

            if label == "Türkçe":
                params["with_original_language"] = "tr"
            else:
                genre_id = genres_dict.get(label.lower())
                if not genre_id: break
                params["with_genres"] = genre_id

            try:
                res = requests.get("https://api.themoviedb.org/3/discover/movie", params=params, timeout=15).json()
                movies = res.get("results", [])
            except Exception as e:
                print(f"📡 Bağlantı hatası: {e}. Bekleniyor...")
                time.sleep(5)
                continue

            if not movies: break

            for m in movies:
                if count >= goal: break

                # m['id'] ve m['original_language'] verilerini kullanarak detay çek ve kaydet
                if save_to_supabase(m['id'], m.get('original_language')):
                    count += 1
                    print(f"✅ {count}/{goal}: {m.get('title')}")

                time.sleep(0.3) # Rate limit koruması
            page += 1

if __name__ == "__main__":
    run_pipeline()
