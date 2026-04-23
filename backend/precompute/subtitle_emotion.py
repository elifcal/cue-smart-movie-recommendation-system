"""
subtitle_emotion.py  (v2 — fused)
==================================
Emotion curve'ü 3 kaynaktan hesaplar ve birleştirir:
  1. Altyazı  (Subdl → HuggingFace duygu modeli)  ağırlık: W_SUB
  2. Ses      (YouTube → librosa RMS)               ağırlık: W_AUD
  3. Görsel   (TMDB poster → renk/parlaklık)        ağırlık: W_VIS

Her kaynak 0-1 aralığında normalize edilmiş 10 elemanlı liste üretir.
Ağırlıklı ortalama → film_dna.emotion_curve[10]

Kaldığı yerden devam eder (islenen_filmler.json).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from dotenv import load_dotenv
from supabase import create_client
from transformers import pipeline

# ── audio.py bağımlılıkları ──────────────────
import librosa

# ── visual.py bağımlılıkları ─────────────────
import colorsys
from io import BytesIO
from colorthief import ColorThief

load_dotenv()

# ─────────────────────────────────────────────
# Sabitler & ağırlıklar
# ─────────────────────────────────────────────
W_SUB = 0.45   # altyazı ağırlığı
W_AUD = 0.35   # ses ağırlığı
W_VIS = 0.20   # görsel ağırlığı

N_DILIM = 10   # emotion_curve uzunluğu

SUBDL_API_KEY    = os.environ.get("SUBDL_API_KEY")
TMDB_API_KEY     = os.environ.get("TMDB_API_KEY")
SUBDL_SEARCH     = "https://api.subdl.com/api/v1/subtitles"
SUBDL_DL_BASE    = "https://dl.subdl.com"
TMDB_IMAGE_BASE  = "https://image.tmdb.org/t/p/w185"
TEST_TABLO       = "film_dna"
ILERLEME_DOSYASI = "islenen_filmler.json"

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

logging.basicConfig(
    filename="errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_sentiment = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)

DIL_MAP = {
    "tr": "TR", "en": "EN", "fr": "FR", "de": "DE",
    "es": "ES", "it": "IT", "pt": "PT", "ru": "RU",
    "ja": "JA", "ko": "KO", "zh": "ZH", "ar": "AR",
    "nl": "NL", "pl": "PL", "sv": "SV", "da": "DA",
}


# ═══════════════════════════════════════════════
# YARDIMCI: İlerleme dosyası
# ═══════════════════════════════════════════════

def islenen_filmleri_yukle() -> set:
    if os.path.exists(ILERLEME_DOSYASI):
        with open(ILERLEME_DOSYASI, "r") as f:
            return set(json.load(f))
    return set()


def islendi_kaydet(tmdb_id: int, islenenler: set):
    islenenler.add(tmdb_id)
    with open(ILERLEME_DOSYASI, "w") as f:
        json.dump(list(islenenler), f)


# ═══════════════════════════════════════════════
# YARDIMCI: normalize — liste 0-1 aralığına çek
# ═══════════════════════════════════════════════

def _normalize(dizi: list[float]) -> list[float]:
    """Min-max normalizasyon; tüm değerler eşitse 0.5 döner."""
    arr = np.array(dizi, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return [0.5] * len(dizi)
    return list(((arr - lo) / (hi - lo)).round(4))


# ═══════════════════════════════════════════════
# KATMAN 1: ALTYAZI  (orijinal subtitle_emotion mantığı)
# ═══════════════════════════════════════════════

def _film_dilini_al(tmdb_id: int) -> str:
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=10,
        )
        if r.ok:
            iso_dil = r.json().get("original_language", "en")
            return DIL_MAP.get(iso_dil, "EN")
    except Exception:
        pass
    return "EN"


def _srt_indir_dil(tmdb_id: int, dil: str) -> str | None:
    try:
        r = requests.get(
            SUBDL_SEARCH,
            params={
                "api_key":       SUBDL_API_KEY,
                "tmdb_id":       tmdb_id,
                "type":          "movie",
                "languages":     dil,
                "subs_per_page": 3,
            },
            timeout=10,
        )
        if not r.ok:
            return None
        subtitles = r.json().get("subtitles", [])
        if not subtitles:
            return None
        en_iyi = sorted(subtitles, key=lambda x: x.get("downloads", 0), reverse=True)[0]
        zip_url = SUBDL_DL_BASE + en_iyi["url"]
        zip_r = requests.get(zip_url, timeout=15)
        if not zip_r.ok:
            return None
        with zipfile.ZipFile(io.BytesIO(zip_r.content)) as z:
            srt_dosyalari = [f for f in z.namelist() if f.endswith(".srt")]
            if not srt_dosyalari:
                return None
            with z.open(srt_dosyalari[0]) as f:
                return f.read().decode("utf-8", errors="ignore")
    except Exception as e:
        logging.error(f"srt_indir hata ({tmdb_id}, {dil}): {e}")
        return None


def _srt_indir(tmdb_id: int, dil: str) -> str | None:
    srt = _srt_indir_dil(tmdb_id, "EN")
    if srt:
        return srt
    if dil != "EN":
        return _srt_indir_dil(tmdb_id, dil)
    return None


def _srt_to_parcalar(srt: str, n: int = N_DILIM) -> list[str]:
    temiz = re.sub(r"\d+\n[\d:,]+ --> [\d:,]+\n", "", srt)
    temiz = re.sub(r"\n{2,}", " ", temiz).strip()
    kelimeler = temiz.split()
    if not kelimeler:
        return [""] * n
    boyut = max(1, len(kelimeler) // n)
    parcalar = []
    for i in range(n):
        bas = i * boyut
        bit = bas + boyut if i < n - 1 else len(kelimeler)
        parcalar.append(" ".join(kelimeler[bas:bit]))
    return parcalar


_DUYGU_AGIRLIKLARI = {
    "fear":     1.0,
    "anger":    0.9,
    "surprise": 0.8,
    "disgust":  0.7,
    "sadness":  0.6,
    "joy":      0.4,
    "neutral":  0.1,
}


def _sentiment_skor(metin: str) -> float:
    if not metin.strip():
        return 0.0
    try:
        sonuclar = _sentiment(metin[:512])[0]
        skor = sum(
            r["score"] * _DUYGU_AGIRLIKLARI.get(r["label"], 0.5)
            for r in sonuclar
        )
        return round(min(skor, 1.0), 4)
    except Exception:
        return 0.0


def subtitle_curve(tmdb_id: int) -> list[float] | None:
    """
    Altyazıdan 10 elemanlı emotion_curve.
    Döner: normalize edilmiş [float]*10  ya da None.
    """
    dil = _film_dilini_al(tmdb_id)
    srt = _srt_indir(tmdb_id, dil)
    if not srt:
        return None
    parcalar = _srt_to_parcalar(srt)
    ham = [_sentiment_skor(p) for p in parcalar]
    return _normalize(ham)


# ═══════════════════════════════════════════════
# KATMAN 2: SES  (librosa RMS tabanlı)
# ═══════════════════════════════════════════════

def _youtube_key_bul(tmdb_id: int) -> str | None:
    """
    TMDB /videos → YouTube trailer key.
    İlk 'Trailer' tipini tercih eder, yoksa ilk videoyu alır.
    """
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos",
            params={"api_key": TMDB_API_KEY},
            timeout=10,
        )
        if not r.ok:
            return None
        sonuclar = r.json().get("results", [])
        youtube = [
            v for v in sonuclar
            if v.get("site") == "YouTube" and v.get("type") == "Trailer"
        ]
        if not youtube:
            youtube = [v for v in sonuclar if v.get("site") == "YouTube"]
        return youtube[0]["key"] if youtube else None
    except Exception as e:
        logging.error(f"youtube_key_bul hata ({tmdb_id}): {e}")
        return None


def _ses_indir(youtube_key: str, klasor: str = "./temp_audio") -> str | None:
    """
    yt-dlp ile ses indir → mp3 yolu.
    yt-dlp kurulu değilse None döner (sessizce).
    """
    try:
        import yt_dlp
        import imageio_ffmpeg
        import shutil

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        hedef = os.path.join(ffmpeg_dir, "ffmpeg")
        if not os.path.exists(hedef):
            shutil.copy(ffmpeg_exe, hedef)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

        os.makedirs(klasor, exist_ok=True)
        mp3_yolu = os.path.join(klasor, f"{youtube_key}.mp3")
        if os.path.exists(mp3_yolu):          # önbellekte varsa tekrar indirme
            return mp3_yolu

        cookie_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")
        secenekler = {
            "format": "bestaudio/best",
            "outtmpl": f"{klasor}/%(id)s.%(ext)s",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "ffmpeg_location": ffmpeg_exe,
            "quiet": True,
            "cookiefile": cookie_path if os.path.exists(cookie_path) else None,
            "nocheckcertificate": True,
            "ignoreerrors": True,
            "no_warnings": True,
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
        }
        with yt_dlp.YoutubeDL(secenekler) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={youtube_key}"])

        return mp3_yolu if os.path.exists(mp3_yolu) else None
    except Exception as e:
        logging.error(f"ses_indir hata ({youtube_key}): {e}")
        return None


def audio_curve(tmdb_id: int) -> list[float] | None:
    """
    YouTube trailer sesinden librosa RMS → 10 dilim → normalize.
    yt-dlp veya ses yoksa None döner.
    """
    key = _youtube_key_bul(tmdb_id)
    if not key:
        return None

    mp3 = _ses_indir(key)
    if not mp3:
        return None

    try:
        y, sr = librosa.load(mp3, duration=180)   # ilk 3 dakika yeterli
        parcalar = np.array_split(y, N_DILIM)
        ham = [float(np.mean(librosa.feature.rms(y=p))) for p in parcalar]
        return _normalize(ham)
    except Exception as e:
        logging.error(f"audio_curve librosa hata ({tmdb_id}): {e}")
        return None


# ═══════════════════════════════════════════════
# KATMAN 3: GÖRSEL  (TMDB poster → renk metrikleri)
# ═══════════════════════════════════════════════

def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h * 360, s, v


def _gorsel_skor(tmdb_id: int) -> float | None:
    """
    Poster → 0-1 arası görsel "yoğunluk" skoru.
    Parlaklık + doygunluk + sıcaklık bileşimi.
    """
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=10,
        )
        if not r.ok:
            return None
        poster_path = r.json().get("poster_path")
        if not poster_path:
            return None

        poster_url = TMDB_IMAGE_BASE + poster_path
        img_r = requests.get(poster_url, timeout=10)
        img_r.raise_for_status()

        ct = ColorThief(BytesIO(img_r.content))
        palette = ct.get_palette(color_count=5, quality=1)

        hsv_list = [_rgb_to_hsv(*c) for c in palette]
        brightness   = np.mean([v for _, _, v in hsv_list])
        saturation   = np.mean([s for _, s, _ in hsv_list])
        warmth_vals  = []
        for h, s, v in hsv_list:
            dist   = min(abs(h - 30), abs(h - 30 + 360), abs(h - 30 - 360))
            warmth_vals.append(max(0.0, 1.0 - dist / 180.0))
        warmth = np.mean(warmth_vals)

        # Bileşik skor: parlak + doygun + sıcak renkler → yüksek görsel enerji
        skor = 0.4 * float(brightness) + 0.35 * float(saturation) + 0.25 * float(warmth)
        return round(min(skor, 1.0), 4)
    except Exception as e:
        logging.error(f"gorsel_skor hata ({tmdb_id}): {e}")
        return None


def visual_curve(tmdb_id: int) -> list[float] | None:
    """
    Poster'dan tek bir skor üretilir; tüm 10 dilime eşit dağıtılır.
    Poster stil filmin geneline yansıdığından sabit tonal zemin mantıklıdır.
    Döner: [skor]*10  ya da None.
    """
    skor = _gorsel_skor(tmdb_id)
    if skor is None:
        return None
    return [skor] * N_DILIM


# ═══════════════════════════════════════════════
# BİRLEŞTİRİCİ: 3 katman → fused emotion_curve
# ═══════════════════════════════════════════════

def fused_emotion_curve(tmdb_id: int) -> list[float] | None:
    """
    Üç kaynağı ağırlıklı olarak birleştirir.

    Eksik kaynak varsa ağırlıklar mevcut kaynaklar arasında yeniden bölünür,
    böylece hiç altyazı olmasa bile ses + görsel ile çalışmaya devam eder.
    """
    kaynak_sonuclari: dict[str, list[float] | None] = {
        "sub": subtitle_curve(tmdb_id),
        "aud": audio_curve(tmdb_id),
        "vis": visual_curve(tmdb_id),
    }
    agirliklar = {"sub": W_SUB, "aud": W_AUD, "vis": W_VIS}

    mevcut = {k: v for k, v in kaynak_sonuclari.items() if v is not None}
    if not mevcut:
        return None

    # Eksik kaynakların ağırlıklarını mevcutlara orantılı dağıt
    toplam_agirlik = sum(agirliklar[k] for k in mevcut)
    olcekli = {k: agirliklar[k] / toplam_agirlik for k in mevcut}

    fused = np.zeros(N_DILIM, dtype=float)
    for k, curve in mevcut.items():
        fused += olcekli[k] * np.array(curve)

    return [round(float(x), 4) for x in fused]


# ═══════════════════════════════════════════════
# TOPLU GÜNCELLEME
# ═══════════════════════════════════════════════

def toplu_guncelle(max_workers: int = 5, limit: int = None, offset: int = 0):
    """
    offset : kaçıncı filmden başlanacak (0 = baştan)
    limit  : kaç film işlenecek (None = offset'ten sona kadar hepsi)

    Kullanım örnekleri:
      toplu_guncelle()                        → tüm filmler
      toplu_guncelle(offset=0,    limit=1000) → 1-1000
      toplu_guncelle(offset=1000, limit=1000) → 1001-2000
      toplu_guncelle(offset=2000)             → 2001-son
    """
    filmler = []
    sayfa = 0
    sayfa_boyutu = 1000
    while True:
        sorgu = (
            supabase.table(TEST_TABLO)
            .select("tmdb_id, title")
            .range(sayfa * sayfa_boyutu, (sayfa + 1) * sayfa_boyutu - 1)
        )
        sonuc = sorgu.execute().data
        if not sonuc:
            break
        filmler.extend(sonuc)
        if len(sonuc) < sayfa_boyutu:
            break
        sayfa += 1

    # offset + limit uygula
    filmler = filmler[offset:]
    if limit:
        filmler = filmler[:limit]

    islenenler = islenen_filmleri_yukle()
    kalan = [f for f in filmler if f["tmdb_id"] not in islenenler]

    print(f"{'=' * 50}")
    print(f"TABLO      : {TEST_TABLO}")
    print(f"OFFSET     : {offset}  →  LIMIT: {limit if limit else 'sona kadar'}")
    print(f"BU GRUPTA  : {len(filmler)} film")
    print(f"DAHA ÖNCE  : {len(islenenler)} film işlendi")
    print(f"İŞLENECEK  : {len(kalan)} film")
    print(f"WORKERS    : {max_workers}")
    print(f"AĞIRLIKLAR : sub={W_SUB}  aud={W_AUD}  vis={W_VIS}")
    print(f"{'=' * 50}\n")

    basarili = 0
    basarisiz = 0
    atlandi   = 0

    def isle(film):
        nonlocal basarili, basarisiz, atlandi
        tmdb_id = film["tmdb_id"]
        title   = film.get("title", "?")
        try:
            curve = fused_emotion_curve(tmdb_id)
            if curve:
                supabase.table(TEST_TABLO).update(
                    {"emotion_curve": curve}
                ).eq("tmdb_id", tmdb_id).execute()
                print(f"✅ [{tmdb_id}] {title}")
                islendi_kaydet(tmdb_id, islenenler)
                basarili += 1
            else:
                print(f"⚠️  [{tmdb_id}] {title} — hiçbir kaynak bulunamadı")
                islendi_kaydet(tmdb_id, islenenler)
                atlandi += 1
        except Exception as e:
            logging.error(f"isle hata ({tmdb_id}): {e}")
            print(f"❌ [{tmdb_id}] {title} — {e}")
            basarisiz += 1
        time.sleep(0.2)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        ex.map(isle, kalan)

    print(f"\n{'=' * 50}")
    print(f"✅  Başarılı       : {basarili}")
    print(f"⚠️   Kaynak yok    : {atlandi}")
    print(f"❌  Hata           : {basarisiz}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # ── Batch ayarları ──────────────────────────────
    # 1. çalıştırma (API key 1):  offset=0,    limit=1000
    # 2. çalıştırma (API key 2):  offset=1000, limit=1000
    # 3. çalıştırma (API key 3):  offset=2000, limit=None
    # ────────────────────────────────────────────────
    toplu_guncelle(max_workers=5, offset=1000, limit=1000)
