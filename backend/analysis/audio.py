import os
import shutil
import subprocess
import imageio_ffmpeg
import yt_dlp
import whisper
import librosa
import numpy as np

# FFmpeg PATH ayarı
def ffmpeg_ayarla():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    hedef = os.path.join(ffmpeg_dir, "ffmpeg")
    if not os.path.exists(hedef):
        shutil.copy(ffmpeg_exe, hedef)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

ffmpeg_ayarla()

# Ses indirme
def ses_indir(youtube_key, cikti_klasor="./temp_audio"):
    os.makedirs(cikti_klasor, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={youtube_key}"
    secenekler = {
        'format': 'bestaudio/best',
        'outtmpl': f'{cikti_klasor}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(secenekler) as ydl:
        ydl.download([url])
    mp3_yolu = os.path.join(cikti_klasor, f"{youtube_key}.mp3")
    return mp3_yolu if os.path.exists(mp3_yolu) else None

# Whisper transkript
def transkript_cikar(mp3_yolu):
    try:
        model = whisper.load_model("base")
        sonuc = model.transcribe(mp3_yolu)
        return sonuc["text"]
    except Exception as e:
        print(f"❌ Transkript hatası ({mp3_yolu}): {e}")
        return ""

# Tempo hesaplama
def tempo_hesapla(mp3_yolu, sure=60):
    try:
        if mp3_yolu is None:
            return None
        if not os.path.exists(mp3_yolu):
            print(f"⚠️ Dosya bulunamadı: {mp3_yolu}")
            return None
        y, sr = librosa.load(mp3_yolu, duration=sure)
        if len(y) == 0:
            print(f"⚠️ Ses dosyası boş: {mp3_yolu}")
            return None
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return round(float(np.atleast_1d(tempo)[0]), 1)
    except Exception as e:
        print(f"❌ Tempo hatası ({mp3_yolu}): {e}")
        return None

# Enerji hesaplama
def enerji_hesapla(mp3_yolu, sure=60):
    try:
        if mp3_yolu is None:
            return None
        if not os.path.exists(mp3_yolu):
            print(f"⚠️ Dosya bulunamadı: {mp3_yolu}")
            return None
        y, sr = librosa.load(mp3_yolu, duration=sure)
        if len(y) == 0:
            print(f"⚠️ Ses dosyası boş: {mp3_yolu}")
            return None
        enerji = float(np.mean(librosa.feature.rms(y=y)))
        return round(enerji, 4)
    except Exception as e:
        print(f"❌ Enerji hatası ({mp3_yolu}): {e}")
        return None

# Duygu eğrisi (10 noktalı)
def duygu_egrisi_cikar(mp3_yolu, sure=60):
    try:
        if mp3_yolu is None:
            return None
        y, sr = librosa.load(mp3_yolu, duration=sure)
        parcalar = np.array_split(y, 10)
        egri = [float(np.mean(librosa.feature.rms(y=p))) for p in parcalar]
        return egri
    except Exception as e:
        print(f"❌ Duygu eğrisi hatası ({mp3_yolu}): {e}")
        return None

# Ses kalitesi kontrol
def ses_kalitesi_kontrol(mp3_yolu, min_sure=10, min_enerji=0.001):
    try:
        y, sr = librosa.load(mp3_yolu, duration=60)
        sure = librosa.get_duration(y=y, sr=sr)
        enerji = float(np.mean(librosa.feature.rms(y=y)))
        if sure < min_sure:
            print(f"⚠️ Çok kısa: {sure:.1f} saniye")
            return False
        if enerji < min_enerji:
            print(f"⚠️ Çok sessiz: enerji {enerji}")
            return False
        return True
    except Exception as e:
        print(f"❌ Ses kalitesi kontrol hatası: {e}")
        return False

# Anahtar kelime listesi
ANAHTAR_KELIMELER = {
    "korku":   ["death", "kill", "dark", "fear", "monster", "run", "scream",
                "demon", "hell", "terrifying", "survive", "entity", "banish",
                "ghost", "horror", "evil", "curse", "haunted",
                "ölüm", "karanlık", "korku", "kaç", "canavar"],
    "gerilim": ["secret", "mystery", "danger", "threat", "escape", "trap",
                "freeze", "command", "control",
                "gizem", "tehlike", "tuzak", "kaçış"],
    "heyecan": ["fight", "battle", "war", "explosion", "chase", "hero",
                "savaş", "patlama", "kahraman", "dövüş"],
    "dram":    ["love", "life", "family", "hope", "dream", "loss",
                "aşk", "umut", "aile", "kayıp"],
    "komedi":  ["funny", "laugh", "crazy", "weird", "stupid",
                "komik", "gülünç", "saçma"],
}

def anahtar_kelime_cikar(transkript):
    transkript_lower = transkript.lower()
    bulunanlar = {}
    for kategori, kelimeler in ANAHTAR_KELIMELER.items():
        eslesen = [k for k in kelimeler if k in transkript_lower]
        if eslesen:
            bulunanlar[kategori] = eslesen
    return bulunanlar

def konusma_orani_hesapla(mp3_yolu):
    try:
        model = whisper.load_model("base")
        sonuc = model.transcribe(mp3_yolu, verbose=False)
        segmentler = sonuc.get("segments", [])
        if not segmentler:
            return 0.0
        toplam_sure = segmentler[-1]["end"]
        konusma_suresi = sum(s["end"] - s["start"] for s in segmentler)
        oran = konusma_suresi / toplam_sure if toplam_sure > 0 else 0
        return round(oran, 3)
    except Exception as e:
        print(f"❌ Konuşma oranı hatası ({mp3_yolu}): {e}")
        return 0.0
