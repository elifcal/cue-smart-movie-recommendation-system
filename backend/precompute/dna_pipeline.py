import os
import sys
import time
import json
import whisper
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv

# Proje dizin yapısına göre modülleri içe aktarma
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.audio import ses_indir, analyze_audio
from analysis.visual import analyze_visual

# Ortam değişkenlerini yükleme ve veritabanı bağlantısı
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

print(" Test başlatılıyor ve Supabase'e KAYIT modu aktif...")

def test_dna_isle(film):
    """
    Film verilerini analiz eder ve sonuçları Supabase'deki 'film_dna' tablosuna yazar.
    """
    tmdb_id = film.get("tmdb_id")
    title = film.get("title")
    videos = film.get("videos", [])
    mp3_yolu = None

    print(f"\n--- {title} İşleniyor ---")

    if not videos:
        print(f"Video verisi yok (Boş Liste), atlanıyor.")
        return

    trailer_key = next(
        (
            v.get("key")
            for v in videos
            if isinstance(v, dict)
            and v.get("site") == "YouTube"
            and v.get("type") == "Trailer"
        ),
        None,
    )

    if not trailer_key:
        print(f"Geçerli bir YouTube fragmanı (Trailer) bulunamadı, atlanıyor.")
        return

    youtube_url = f"https://www.youtube.com/watch?v={trailer_key}"
    print(f" Hedef URL: {youtube_url}")

    try:
        # 1. SES İNDİRME
        mp3_yolu = ses_indir(trailer_key)
        if not mp3_yolu or not os.path.exists(mp3_yolu):
            print(f" Video yt-dlp ile indirilemedi.")
            return

        # 2. SES ANALİZİ
        print("   -> Ses analizi başlatılıyor...")
        ses_sonuclari = analyze_audio(mp3_yolu, sure=60)

        if ses_sonuclari is None:
            print(f"Ses analizi başarısız, atlanıyor.")
            return

        # 3. GÖRSEL ANALİZ
        print("Görsel analiz başlatılıyor...")
        poster_url = f"https://image.tmdb.org/t/p/w500{film.get('poster_path')}"
        gorsel_sonuclari = analyze_visual(tmdb_id, poster_url)

        # 4. SUPABASE'E YAZILACAK VERİYİ HAZIRLAMA (Görsellere Birebir Uygun)
        tam_metin = ses_sonuclari.get("text", "")
        
        # O anki zamanı Supabase'in kabul edeceği formatta (ISO 8601) alıyoruz
        su_an = datetime.now(timezone.utc).isoformat()

        dna_verisi = {
            "tmdb_id": tmdb_id,
            "title": title,
            "whisper_text": tam_metin,
            "emotion_curve": ses_sonuclari.get("emotion_curve"),
            "tempo": ses_sonuclari.get("tempo"),
            "speech_ratio": ses_sonuclari.get("speech_ratio"),
            "color_palette": gorsel_sonuclari.get("color_palette"),
            "brightness": gorsel_sonuclari.get("brightness"),
            "warmth": gorsel_sonuclari.get("warmth"),
            "saturation": gorsel_sonuclari.get("saturation"),
            "energy": ses_sonuclari.get("energy"),
            "analyzed_at": su_an  # Yeni eklenen zaman damgası sütunu
        }

        # 5. SUPABASE'E KAYDETME
        print("   -> Veritabanına (film_dna tablosuna) kaydediliyor...")
        response = supabase.table("film_dna").upsert(dna_verisi).execute()
        
        print(f"BAŞARIYLA KAYDEDİLDİ: {title}")

    except Exception as e:
        hata_mesaji = str(e)

        if "Private video" in hata_mesaji:
            print(f"Private video, erişilemiyor — atlanıyor.")
        elif "No supported JavaScript runtime" in hata_mesaji:
            print(f"yt-dlp JS runtime bulunamadı.")
            print(f"   Detay: {hata_mesaji}")
        else:
            print(f"Beklenmeyen hata: {hata_mesaji}")

    finally:
        # 6. TEMİZLİK
        if mp3_yolu and os.path.exists(mp3_yolu):
            try:
                os.remove(mp3_yolu)
                print(f"Geçici dosya temizlendi: {mp3_yolu}")
            except OSError as e:
                print(f"Geçici dosya silinemedi: {e}")

if __name__ == "__main__":
    print("Supabase üzerinden ilk 5 film çekiliyor...")
    res = supabase.table("movies").select("*").range(0, 49).execute()
    filmler = res.data

    if not filmler:
        print("Test edilecek film bulunamadı.")
        sys.exit(0)

    for film in filmler:
        test_dna_isle(film)
        print("-" * 60)
        time.sleep(1)

    print(f"\nİşlem tamamlandı: {len(filmler)} film işlendi ve veritabanına yazıldı.")