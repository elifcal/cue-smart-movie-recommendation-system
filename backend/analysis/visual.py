"visual.py"
import os
import requests
import colorsys
import logging  # ← 1. EKLEME
from io import BytesIO
from colorthief import ColorThief

from dotenv import load_dotenv
load_dotenv()

# ← 2. EKLEME
logging.basicConfig(
    filename="errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"


def renk_paleti_cikar(poster_url: str) -> dict:
    """Poster URL'sinden baskın renk + 5 renkli palet döndürür."""
    try:                                                          # ← 3. EKLEME
        params = {"api_key": TMDB_API_KEY} if "tmdb.org" in poster_url else {}
        response = requests.get(poster_url, params=params, timeout=10)
        response.raise_for_status()
        ct = ColorThief(BytesIO(response.content))
        dominant = ct.get_color(quality=1)
        palette = ct.get_palette(color_count=5, quality=1)
        return {"dominant": dominant, "palette": palette}
    except FileNotFoundError as e:
        logging.error(f"Colorthief - Dosya bulunamadı: {e}")
        print(f"❌ Colorthief hatası: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Colorthief - İstek hatası: {e}")
        print(f"❌ Poster indirilemedi: {e}")
        return None
    except Exception as e:
        logging.error(f"Colorthief beklenmedik hata: {e}")
        print(f"❌ Beklenmedik hata: {e}")
        return None


def rgb_to_hsv(r: int, g: int, b: int) -> tuple:
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return round(h * 360, 1), round(s, 4), round(v, 4)


def parlaklik_hesapla(palette: list) -> float:
    v_degerleri = [rgb_to_hsv(*renk)[2] for renk in palette]
    return round(sum(v_degerleri) / len(v_degerleri), 3)


def doygunluk_hesapla(palette: list) -> float:
    s_degerleri = [rgb_to_hsv(*renk)[1] for renk in palette]
    return round(sum(s_degerleri) / len(s_degerleri), 3)


def sicaklik_hesapla(palette: list) -> float:
    import math
    sicakliklar = []
    for renk in palette:
        h, s, v = rgb_to_hsv(*renk)
        distance = min(abs(h - 30), abs(h - 30 + 360), abs(h - 30 - 360))
        warmth = max(0.0, 1.0 - distance / 180.0)
        sicakliklar.append(warmth)
    return round(sum(sicakliklar) / len(sicakliklar), 3)


def analyze_visual(tmdb_id: int, poster_path: str) -> dict:
    if poster_path.startswith("http"):
        renkler = renk_paleti_cikar(poster_path)
    else:
        try:                                                      # ← 4. EKLEME
            with open(poster_path, "rb") as f:
                ct = ColorThief(f)
                dominant = ct.get_color(quality=1)
                palette = ct.get_palette(color_count=5, quality=1)
            renkler = {"dominant": dominant, "palette": palette}
        except FileNotFoundError as e:
            logging.error(f"Colorthief - Yerel dosya bulunamadı: {e}")
            print(f"❌ Dosya bulunamadı: {e}")
            return None
        except Exception as e:
            logging.error(f"Colorthief - Yerel dosya hatası: {e}")
            print(f"❌ Beklenmedik hata: {e}")
            return None

    if renkler is None:                                           # ← 5. EKLEME
        return None

    palette = renkler["palette"]

    return {
        "film_id":       tmdb_id,
        "dominant_renk": renkler["dominant"],
        "color_palette": [[int(x) for x in r] for r in palette],
        "brightness":    parlaklik_hesapla(palette),
        "saturation":    doygunluk_hesapla(palette),
        "warmth":        sicaklik_hesapla(palette),
    }
