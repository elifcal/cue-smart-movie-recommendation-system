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
    model = whisper.load_model("base")
    sonuc = model.transcribe(mp3_yolu)
    return sonuc["text"]

# Librosa tempo ve enerji
def tempo_enerji_analiz(mp3_yolu, sure=60):
    y, sr = librosa.load(mp3_yolu, duration=sure)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    enerji = float(np.mean(librosa.feature.rms(y=y)))
    return {"tempo_bpm": float(tempo), "enerji": enerji}

# Duygu eğrisi (10 noktalı)
def duygu_egrisi_cikar(mp3_yolu, sure=60):
    y, sr = librosa.load(mp3_yolu, duration=sure)
    parcalar = np.array_split(y, 10)
    egri = [float(np.mean(librosa.feature.rms(y=p))) for p in parcalar]
    return egri

# Konuşma oranı
def konusma_orani(transkript, mp3_yolu, sure=60):
    kelime_sayisi = len(transkript.split())
    oran = min(kelime_sayisi / (sure * 2), 1.0)
    return round(oran, 3)
