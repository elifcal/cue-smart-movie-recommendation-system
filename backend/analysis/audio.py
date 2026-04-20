import os
import shutil
import imageio_ffmpeg
import yt_dlp
import whisper
import librosa
import numpy as np

def ffmpeg_ayarla():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    hedef = os.path.join(ffmpeg_dir, "ffmpeg")
    if not os.path.exists(hedef):
        shutil.copy(ffmpeg_exe, hedef)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

ffmpeg_ayarla()

def ses_indir(youtube_key, cikti_klasor="./temp_audio"):
    os.makedirs(cikti_klasor, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={youtube_key}"
    cookie_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")

    secenekler = {
        'format': 'bestaudio/best',
        'outtmpl': f'{cikti_klasor}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
        'quiet': False,
        'cookiefile': cookie_path if os.path.exists(cookie_path) else None,
        'nocheckcertificate': True,
        'ignoreerrors': False,
        'no_warnings': False,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'referer': 'https://www.google.com/',
    }

    try:
        with yt_dlp.YoutubeDL(secenekler) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"⚠️ İndirme kritik hata: {e}")

    mp3_yolu = os.path.join(cikti_klasor, f"{youtube_key}.mp3")
    return mp3_yolu if os.path.exists(mp3_yolu) else None

def analyze_audio(mp3_yolu, model, sure=60, language=None):
    if mp3_yolu is None or not os.path.exists(mp3_yolu):
        return None

    print(f"🎙 Analiz ediliyor: {mp3_yolu}")
    sonuc = model.transcribe(mp3_yolu, verbose=False, language=language)
    text = sonuc["text"]
    algilanan_dil = sonuc.get("language")
    dil_uyumsuzlugu = None
    if language and algilanan_dil and algilanan_dil != language:
     uyari = f"Dil uyuşmazlığı: TMDB={language}, Whisper={algilanan_dil}"
     print(f"⚠️  {uyari}")
     dil_uyumsuzlugu = uyari
    segmentler = sonuc.get("segments", [])

    y, sr = librosa.load(mp3_yolu, duration=sure)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = round(float(np.atleast_1d(tempo)[0]), 1)
    energy = round(float(np.mean(librosa.feature.rms(y=y))), 4)
    parcalar = np.array_split(y, 10)
    emotion_curve = [float(np.mean(librosa.feature.rms(y=p))) for p in parcalar]

    if segmentler:
     toplam_sure = sure  # segmentler[-1]["end"] yerine sabit sure
     konusma_suresi = sum(s["end"] - s["start"] for s in segmentler)
     speech_ratio = round(min(konusma_suresi / toplam_sure, 1.0), 3)
    else:
        speech_ratio = 0.0

    return {
        "text": text,
        "tempo": tempo,
        "energy": energy,
        "emotion_curve": emotion_curve,
        "speech_ratio": speech_ratio,
        "detected_language": algilanan_dil,
        "dil_uyumsuzlugu": dil_uyumsuzlugu,
    }
