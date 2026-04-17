import requests
from io import BytesIO
from colorthief import ColorThief

# Renk paleti çıkar
def renk_paleti_cikar(poster_url):
    response = requests.get(poster_url)
    ct = ColorThief(BytesIO(response.content))
    dominant = ct.get_color(quality=1)
    palette = ct.get_palette(color_count=5)
    return {"dominant": dominant, "palette": palette}

# Parlaklık (0.0 karanlık → 1.0 aydınlık)
def parlaklik_hesapla(palette):
    ortalama = sum(sum(renk) / 3 for renk in palette) / len(palette)
    return round(ortalama / 255, 3)

# Doygunluk (0.0 soluk → 1.0 canlı)
def doygunluk_hesapla(palette):
    doygunluklar = []
    for r, g, b in palette:
        maksimum = max(r, g, b)
        minimum = min(r, g, b)
        doy = (maksimum - minimum) / maksimum if maksimum != 0 else 0
        doygunluklar.append(doy)
    return round(sum(doygunluklar) / len(doygunluklar), 3)

# Sıcaklık (0.0 soğuk/mavi → 1.0 sıcak/kırmızı)
def sicaklik_hesapla(palette):
    sicakliklar = []
    for r, g, b in palette:
        toplam = r + g + b
        sicaklik = r / toplam if toplam != 0 else 0.33
        sicakliklar.append(sicaklik)
    return round(sum(sicakliklar) / len(sicakliklar), 3)

# Tüm görsel analizi tek fonksiyonda çalıştır
def analyze_visual(tmdb_id, poster_url):
    # tmdb_id parametresi bu aşamada doğrudan görsel analizde kullanılmasa da, 
    # ana boru hattının gönderdiği parametre imzasını (signature) karşılamak zorundadır.
    
    renkler = renk_paleti_cikar(poster_url)
    palette = renkler["palette"]
    
    # Anahtar isimleri test_pipeline.py'nin beklediği formata (İngilizce) çevrildi
    return {
        "dominant_color": renkler["dominant"],
        "color_palette": palette,
        "brightness": parlaklik_hesapla(palette),
        "saturation": doygunluk_hesapla(palette),
        "warmth": sicaklik_hesapla(palette),
    }
