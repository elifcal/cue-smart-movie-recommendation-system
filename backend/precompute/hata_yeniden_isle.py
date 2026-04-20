"""
Birden fazla hata raporu xlsx'inden TMDB ID'leri okur,
Supabase'den o filmlerin verisini çeker ve pipeline'a gönderir.

Kullanım:
    python hata_yeniden_isle.py hata1.xlsx hata2.xlsx ...
    python hata_yeniden_isle.py hata1.xlsx hata2.xlsx --force   # zaten işlenmiş olsalar bile
    python hata_yeniden_isle.py hata1.xlsx --sadece-video-yok   # sadece "key bulunamadı" hataları
    python hata_yeniden_isle.py hata1.xlsx --sadece-metadata    # sadece "metadata alınamadı" hataları
"""

import sys
import os
import argparse
import time
import random
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dna_pipeline import dna_isle, hata_raporu_olustur, hatali_filmler
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def tmdb_idleri_oku(xlsx_yollari, sadece_metadata=False, sadece_video_yok=False):
    """Birden fazla xlsx'den TMDB ID ve hata bilgilerini okur."""
    tum_kayitlar = []

    for yol in xlsx_yollari:
        if not os.path.exists(yol):
            print(f"⚠️  Dosya bulunamadı, atlandı: {yol}")
            continue

        try:
            df = pd.read_excel(yol, skiprows=2)
            df.columns = ["#", "Film Adı", "TMDB ID", "YouTube URL", "Hata Mesajı"]
            df = df.dropna(subset=["TMDB ID"])
            df["TMDB ID"] = df["TMDB ID"].astype(int)
            print(f"📂 {os.path.basename(yol)}: {len(df)} hatalı film okundu")
            tum_kayitlar.append(df)
        except Exception as e:
            print(f"❌ {yol} okunamadı: {e}")

    if not tum_kayitlar:
        print("❌ Hiç geçerli xlsx bulunamadı.")
        return []

    birlesik = pd.concat(tum_kayitlar, ignore_index=True)

    # Filtre uygula
    if sadece_metadata:
        birlesik = birlesik[birlesik["Hata Mesajı"].str.contains("metadata", na=False)]
        print(f"🔍 Filtre: sadece 'metadata' hataları → {len(birlesik)} kayıt")
    elif sadece_video_yok:
        birlesik = birlesik[birlesik["Hata Mesajı"].str.contains("bulunamadı", na=False)]
        print(f"🔍 Filtre: sadece 'video/key bulunamadı' hataları → {len(birlesik)} kayıt")

    # Tekrar eden TMDB ID'leri temizle
    onceki = len(birlesik)
    birlesik = birlesik.drop_duplicates(subset=["TMDB ID"])
    if len(birlesik) < onceki:
        print(f"🔄 {onceki - len(birlesik)} tekrar eden kayıt temizlendi")

    print(f"✅ Toplam işlenecek: {len(birlesik)} film\n")
    return birlesik["TMDB ID"].tolist()


def filmleri_yukle(tmdb_idler):
    """Supabase'den verilen TMDB ID'lerin tam film verisini çeker."""
    filmler = []
    # Supabase'in IN filtresi için chunk'lara böl (max 100)
    for i in range(0, len(tmdb_idler), 100):
        chunk = tmdb_idler[i:i+100]
        res = supabase.table("movies").select("*").in_("tmdb_id", chunk).execute()
        filmler.extend(res.data)

    print(f"📦 Supabase'den {len(filmler)}/{len(tmdb_idler)} film yüklendi")

    # Bulunamayanları raporla
    bulunan_idler = {f["tmdb_id"] for f in filmler}
    eksik = set(tmdb_idler) - bulunan_idler
    if eksik:
        print(f"⚠️  {len(eksik)} film movies tablosunda bulunamadı: {sorted(eksik)}")

    return filmler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hatalı filmleri yeniden işle")
    parser.add_argument("xlsx", nargs="+", help="Hata raporu xlsx dosyaları")
    parser.add_argument("--force", action="store_true", help="Zaten işlenmiş olsalar bile yeniden işle")
    parser.add_argument("--sadece-metadata", action="store_true", help="Sadece 'metadata alınamadı' hatalarını işle")
    parser.add_argument("--sadece-video-yok", action="store_true", help="Sadece 'video/key bulunamadı' hatalarını işle")
    args = parser.parse_args()

    print("=" * 55)
    print("🔁 HATALI FİLMLERİ YENİDEN İŞLE")
    print("=" * 55)

    tmdb_idler = tmdb_idleri_oku(
        args.xlsx,
        sadece_metadata=args.sadece_metadata,
        sadece_video_yok=args.sadece_video_yok,
    )

    if not tmdb_idler:
        print("📭 İşlenecek film bulunamadı.")
        sys.exit(0)

    filmler = filmleri_yukle(tmdb_idler)

    if not filmler:
        print("📭 Supabase'de eşleşen film yok.")
        sys.exit(0)

    toplam = len(filmler)
    baslangic = time.time()

    for i, film in enumerate(filmler):
        print(f"\n📊 İlerleme: {i + 1} / {toplam}")
        dna_isle(film, force=args.force)
        if i < toplam - 1:
            time.sleep(random.randint(1, 3))

    toplam_sure = time.time() - baslangic
    print(f"\n⏱️  Toplam Süre: {int(toplam_sure // 60)} dk {int(toplam_sure % 60)} sn")
    print(f"📈 Ortalama: {round(toplam_sure / toplam, 2)} sn/film")

    hata_raporu_olustur(hatali_filmler, toplam)
