from ml.dna_storage import save_dna_to_supabase

# fake data (test verisi)
ses_verisi = {
    "tempo": 120,
    "energy": 0.8,
    "speech_ratio": 0.3,
    "emotion_curve": [0.1, 0.5, 0.9]
}

gorsel_verisi = {
    "brightness": 0.7,
    "saturation": 0.6,
    "warmth": 0.5
}

result = save_dna_to_supabase(
    tmdb_id=999999,
    title="TEST MOVIE",
    whisper_text="This is a test",
    ses_verisi=ses_verisi,
    gorsel_verisi=gorsel_verisi
)

print(result)
