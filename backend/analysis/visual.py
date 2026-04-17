import os
import requests
import colorsys
from io import BytesIO
from colorthief import ColorThief

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"


# ─────────────────────────────────────────────
# Renk paleti çıkar (ColorThief)
# ─────────────────────────────────────────────
def renk_paleti_cikar(poster_url: str) -> dict:
    """Poster URL'sinden baskın renk + 5 renkli palet döndürür."""
    params = {"api_key": TMDB_API_KEY} if "tmdb.org" in poster_url else {}
    response = requests.get(poster_url, params=params, timeout=10)
    response.raise_for_status()
    ct = ColorThief(BytesIO(response.content))
    dominant = ct.get_color(quality=1)
    palette = ct.get_palette(color_count=5, quality=1)
    return {"dominant": dominant, "palette": palette}


# ─────────────────────────────────────────────
# RGB → HSV dönüşümü yardımcısı
# ─────────────────────────────────────────────
def rgb_to_hsv(r: int, g: int, b: int) -> tuple:
    """0-255 RGB → (h 0-360, s 0-1, v 0-1) döndürür."""
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return round(h * 360, 1), round(s, 4), round(v, 4)


# ─────────────────────────────────────────────
# Parlaklık  (0.0 karanlık → 1.0 aydınlık)
# HSV'nin V kanalının palet ortalaması
# ─────────────────────────────────────────────
def parlaklik_hesapla(palette: list) -> float:
    v_degerleri = [rgb_to_hsv(*renk)[2] for renk in palette]
    return round(sum(v_degerleri) / len(v_degerleri), 3)


# ─────────────────────────────────────────────
# Doygunluk  (0.0 soluk → 1.0 canlı)
# HSV'nin S kanalının palet ortalaması
# ─────────────────────────────────────────────
def doygunluk_hesapla(palette: list) -> float:
    s_degerleri = [rgb_to_hsv(*renk)[1] for renk in palette]
    return round(sum(s_degerleri) / len(s_degerleri), 3)


# ─────────────────────────────────────────────
# Sıcaklık  (0.0 soğuk/mavi → 1.0 sıcak/kırmızı)
# Hue'ya göre: kırmızı/turuncu/sarı → sıcak, mavi/mor → soğuk
# ─────────────────────────────────────────────
def sicaklik_hesapla(palette: list) -> float:
    """
    Hue tabanlı sıcaklık.
    0-60° (kırmızı-sarı) → sıcak (1.0)
    180-270° (mavi-yeşil) → soğuk (0.0)
    Formül: cos mesafesi 30°'ye (en sıcak nokta)
    """
    import math
    sicakliklar = []
    for renk in palette:
        h, s, v = rgb_to_hsv(*renk)
        # Kırmızı (0°/360°) ve sarı (60°) arasına yakınlık
        distance = min(abs(h - 30), abs(h - 30 + 360), abs(h - 30 - 360))
        # 0-180 aralığını 0-1'e normalize et (180° tam zıt = soğuk)
        warmth = max(0.0, 1.0 - distance / 180.0)
        sicakliklar.append(warmth)
    return round(sum(sicakliklar) / len(sicakliklar), 3)


# ─────────────────────────────────────────────
# Ana fonksiyon: analyze_visual
# ─────────────────────────────────────────────
def analyze_visual(tmdb_id: int, poster_path: str) -> dict:
    """
    TMDB poster URL'si veya yerel dosya yolundan görsel analiz yapar.

    Parametreler
    ------------
    tmdb_id    : TMDB film ID'si (metadata için)
    poster_path: Tam poster URL'si veya '/path/to/poster.jpg'

    Döndürür
    --------
    dict: film_id, palette, dominant_color,
          brightness, saturation, warmth
    """
    # URL mi, yerel dosya mı?
    if poster_path.startswith("http"):
        renkler = renk_paleti_cikar(poster_path)
    else:
        # Yerel dosya
        with open(poster_path, "rb") as f:
            ct = ColorThief(f)
            dominant = ct.get_color(quality=1)
            palette = ct.get_palette(color_count=5, quality=1)
        renkler = {"dominant": dominant, "palette": palette}

    palette = renkler["palette"]

    return {
        "film_id":       tmdb_id,
        "dominant_renk": renkler["dominant"],
        "renk_paleti":   palette,
        "brightness":    parlaklik_hesapla(palette),
        "saturation":    doygunluk_hesapla(palette),
        "warmth":        sicaklik_hesapla(palette),
    }


# ─────────────────────────────────────────────
# Tür-Renk Korelasyon Testi (10 Film)
# ─────────────────────────────────────────────
TEST_FILMLERI = [
    # (tmdb_id, tur, poster_url)
    # --- KORKU ---
    (694,   "korku",   "https://image.tmdb.org/t/p/w185/qV9cA4G26FaFYAIRFWKPdF1XpNe.jpg"),  # The Shining
    (539,   "korku",   "https://image.tmdb.org/t/p/w185/rSFM0TbpViSNQQJNxHBE0V5vQkT.jpg"),  # Psycho
    (348,   "korku",   "https://image.tmdb.org/t/p/w185/9O7gLzmreU0nGkIB6K3BsJbzvNv.jpg"),  # Alien
    # --- AKSİYON ---
    (155,   "aksiyon", "https://image.tmdb.org/t/p/w185/qJ2tW6WMUDux911r6m7haRef0WH.jpg"),  # The Dark Knight
    (24428, "aksiyon", "https://image.tmdb.org/t/p/w185/cezWGskPY5x7GaglTTRN4Fugfb8.jpg"),  # The Avengers
    (157336,"aksiyon", "https://image.tmdb.org/t/p/w185/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg"),  # Interstellar
    # --- KOMEDİ ---
    (120,   "komedi",  "https://image.tmdb.org/t/p/w185/56v2KjBlU4XaOv9rVYEQypROD7P.jpg"),  # The Lord of the Rings (fantasy/adventure - using as proxy)
    (13,    "komedi",  "https://image.tmdb.org/t/p/w185/saHP97rTPS5eLmrLQEcANmKrsFl.jpg"),  # Forrest Gump
    # --- DRAM ---
    (550,   "dram",    "https://image.tmdb.org/t/p/w185/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"),  # Fight Club
    (238,   "dram",    "https://image.tmdb.org/t/p/w185/3bhkrj58Vtu7enYsLlegkASewhW.jpg"),  # The Godfather
]


def tur_renk_korelasyon_test():
    """
    10 film üzerinde görsel analiz yapıp tür bazlı parlaklık
    ortalamasını hesaplar. Korku filmleri gerçekten daha koyu mu?
    """
    print("=" * 60)
    print("TÜR-RENK KORELASYON ANALİZİ")
    print("=" * 60)

    sonuclar = []
    for tmdb_id, tur, url in TEST_FILMLERI:
        try:
            analiz = analyze_visual(tmdb_id, url)
            analiz["tur"] = tur
            sonuclar.append(analiz)
            print(f"✅ [{tur:8}] film_id={tmdb_id:6} | "
                  f"brightness={analiz['brightness']:.3f} | "
                  f"saturation={analiz['saturation']:.3f} | "
                  f"warmth={analiz['warmth']:.3f}")
        except Exception as e:
            print(f"❌ film_id={tmdb_id} hata: {e}")

    # Tür bazlı ortalamalar
    print("\n" + "=" * 60)
    print("TÜR BAZLI ORTALAMALAR")
    print("=" * 60)

    turler = {}
    for s in sonuclar:
        t = s["tur"]
        turler.setdefault(t, []).append(s)

    tur_ozetleri = {}
    for tur, filmler in turler.items():
        ort_brightness = sum(f["brightness"] for f in filmler) / len(filmler)
        ort_saturation = sum(f["saturation"] for f in filmler) / len(filmler)
        ort_warmth     = sum(f["warmth"]     for f in filmler) / len(filmler)
        tur_ozetleri[tur] = {
            "brightness": round(ort_brightness, 3),
            "saturation": round(ort_saturation, 3),
            "warmth":     round(ort_warmth,     3),
            "film_sayisi": len(filmler),
        }
        print(f"  {tur:8} ({len(filmler)} film) → "
              f"brightness={ort_brightness:.3f} | "
              f"saturation={ort_saturation:.3f} | "
              f"warmth={ort_warmth:.3f}")

    # Korku vs diğerleri karşılaştırması
    print("\n" + "=" * 60)
    print("KORKU vs DİĞER TÜRLER — BULGULAR")
    print("=" * 60)

    if "korku" in tur_ozetleri:
        korku_brightness = tur_ozetleri["korku"]["brightness"]
        diger_brightness = [
            v["brightness"] for k, v in tur_ozetleri.items() if k != "korku"
        ]
        diger_ort = sum(diger_brightness) / len(diger_brightness)

        print(f"  Korku filmleri ort. parlaklık  : {korku_brightness:.3f}")
        print(f"  Diğer türler ort. parlaklık    : {diger_ort:.3f}")
        if korku_brightness < diger_ort:
            fark = round((diger_ort - korku_brightness) / diger_ort * 100, 1)
            print(f"  ✅ SONUÇ: Korku filmleri %{fark} daha KOYU")
        else:
            fark = round((korku_brightness - diger_ort) / diger_ort * 100, 1)
            print(f"  ⚠️  SONUÇ: Korku filmleri %{fark} daha AYDINLIK (beklenmedik!)")

    return sonuclar, tur_ozetleri


# ─────────────────────────────────────────────
# Script olarak çalıştırılırsa test et
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sonuclar, ozetler = tur_renk_korelasyon_test()
