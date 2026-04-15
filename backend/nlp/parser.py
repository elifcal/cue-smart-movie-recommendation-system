import spacy
from transformers import pipeline

print("Modeller yükleniyor...")
nlp = spacy.blank("tr")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="savasy/bert-base-turkish-sentiment-cased"
)
print("Modeller hazır!\n")

# ---------------------------
# TEMEL EŞLEŞTİRME SÖZLÜKLERİ
# ---------------------------

GENRE_MAP = {
    "korku": 27,
    "aksiyon": 28,
    "komedi": 35,
    "dram": 18,
    "bilim kurgu": 878,
    "gerilim": 53,
    "animasyon": 16,
    "macera": 12,
    "romantik": 10749,
    "suç": 80,
    "gizem": 9648,
    "aile": 10751,
    "fantastik": 14,
    "belgesel": 99,
}

GENRE_SYNONYMS = {
    "korkutucu": "korku",
    "komik": "komedi",
    "heyecanlı": "aksiyon",
    "romcom": "romantik",
}

DECADE_MAP = {
    "60'lar": {"year_gte": 1960, "year_lte": 1969},
    "70'ler": {"year_gte": 1970, "year_lte": 1979},
    "80'ler": {"year_gte": 1980, "year_lte": 1989},
    "90'lar": {"year_gte": 1990, "year_lte": 1999},
    "2000'ler": {"year_gte": 2000, "year_lte": 2009},
    "2010'lar": {"year_gte": 2010, "year_lte": 2019},
    "2020'ler": {"year_gte": 2020, "year_lte": 2029},
    "60larda": {"year_gte": 1960, "year_lte": 1969},
    "70lerde": {"year_gte": 1970, "year_lte": 1979},
    "80lerde": {"year_gte": 1980, "year_lte": 1989},
    "90larda": {"year_gte": 1990, "year_lte": 1999},
    "2000lerde": {"year_gte": 2000, "year_lte": 2009},
    "2010larda": {"year_gte": 2010, "year_lte": 2019},
    "2020lerde": {"year_gte": 2020, "year_lte": 2029},
    "klasik": {"year_gte": 1960, "year_lte": 1999},
    "eski film": {"year_gte": 1950, "year_lte": 1999},
    "yeni film": {"year_gte": 2020, "year_lte": 2029},
    "son yıllardan": {"year_gte": 2020, "year_lte": 2029},
    "güncel": {"year_gte": 2020, "year_lte": 2029},
}

THEME_MAP = {
    "twist": "twist",
    "twistli": "twist",
    "sürpriz son": "twist",
    "şaşırtıcı son": "twist",
    "beklenmedik son": "twist",
    "uzay": "space",
    "uzayda": "space",
    "galaksi": "space",
    "zombi": "zombie",
    "zombie": "zombie",
    "seri katil": "serial_killer",
    "seri katilli": "serial_killer",
    "zaman yolculuğu": "time_travel",
    "distopik": "dystopian",
    "distopya": "dystopian",
    "psikolojik": "psychological",
    "doğaüstü": "supernatural",
    "hayalet": "supernatural",
    "vampir": "vampire",
    "canavar": "monster",
    "robot": "robot",
    "yapay zeka": "ai",
    "intikam": "revenge",
    "hayatta kalma": "survival",
    "savaş": "war",
    "tarihi": "historical",
    "biyografi": "biography",
}

MOOD_KEYWORDS = {
    "karanlık": "dark",
    "kasvetli": "dark",
    "bunaltıcı": "dark",
    "depresif": "dark",
    "gergin": "tense",
    "gerilimli": "tense",
    "üzücü": "sad",
    "ağlatacak": "sad",
    "hüzünlü": "sad",
    "duygusal": "emotional",
    "dokunaklı": "emotional",
    "eğlenceli": "fun",
    "neşeli": "fun",
    "romantik": "romantic",
    "romantizm": "romantic",
    "aşk": "romantic",
    "rahatlatıcı": "light",
    "hafif": "light",
    "gizemli": "mysterious",
    "epik": "epic",
}

STRONG_MOOD_PATTERNS = {
    "dark": [
        "içimi karartacak",
        "çok karanlık",
        "karanlık atmosferli",
        "bunaltıcı",
        "depresif",
    ],
    "light": [
        "rahatlatıcı",
        "hafif bir şey",
        "iç açıcı",
        "kafa dağıtmalık",
        "kafamı dağıtacak",
        "çerezlik",
        "içimi ısıtacak",
        "tatlı bir",
        "tatlı bir şey",
        "keyifli",
        "pozitif",
    ],
    "sad": [
        "ağlatacak",
        "çok üzücü",
        "hüzünlü",
    ],
    "fun": [
        "çok eğlenceli",
        "komik",
        "neşeli",
        "kahkaha atmak istiyorum",
    ],
    "tense": [
        "yüksek tansiyonlu",
        "çok gergin",
        "gerilimli",
        "beni gerecek",
        "gerecek bir",
        "gerecek film",
        "nefes kesen",
        "koltuğuma çivileyecek",
        "koltuğa çivileyecek",
    ],
    "emotional": [
        "çok duygusal",
        "dokunaklı",
        "içli",
    ],
    "romantic": [
        "aşk dolu",
        "romantik",
        "romantizm",
    ],
}

NEGATIVE_MOOD_PATTERNS = {
    "dark": [
        "karanlık olmasın",
        "çok karanlık olmasın",
        "karanlık olmayan",
        "fazla karanlık olmayan",
        "depresif olmasın",
        "depresif olmayan",
    ],
    "tense": [
        "gergin olmasın",
        "çok gergin olmayan",
        "fazla gergin olmayan",
        "gergin olmayan",
        "gerilimli olmayan",
    ],
    "sad": [
        "üzücü olmasın",
        "üzücü olmayan",
        "ağlatmasın",
        "hüzünlü olmasın",
    ],
    "fun": [
        "eğlenceli olmasın",
        "komik olmasın",
        "neşeli olmasın",
    ],
    "romantic": [
        "romantik olmasın",
        "romantik olmayan",
        "romantizm olmasın",
        "romantizm olmasın.",
        "aşk olmasın",
    ],
    "emotional": [
        "duygusal olmasın",
        "duygusal olmayan",
        "dokunaklı olmasın",
    ],
    "light": [
        "hafif olmasın",
        "rahatlatıcı olmasın",
    ],
}

LOW_VIOLENCE_PATTERNS = [
    "az kanlı",
    "az şiddetli",
    "çok şiddetli olmayan",
    "şiddeti düşük",
    "çok kanlı olmayan",
    "çok kanlı olmasın",
    "fazla kanlı olmayan",
    "aşırı sert olmayan",
    "sert olmayan",
    "yumuşak içerikli",
    "aile dostu",
    "kan gövdeyi götürmesin",
]

HIGH_VIOLENCE_PATTERNS = [
    "çok kanlı",
    "aşırı şiddetli",
    "vahşi",
    "kan dolu",
    "sert aksiyon",
]

RUNTIME_PATTERNS = {
    "short": ["kısa film", "çok uzun olmayan", "90 dakikalık", "kısa süreli", "kısa bir film"],
    "long": ["uzun film", "çok uzun", "2 saatten uzun", "uzun süreli", "uzun soluklu", "epik"],
}

LANGUAGE_MAP = {
    "türkçe": "tr",
    "ingilizce": "en",
    "korece": "ko",
    "fransızca": "fr",
    "japonca": "ja",
    "ispanyolca": "es",
    "italyanca": "it",
    "almanca": "de",
    "çince": "zh",
    "hintçe": "hi",
}

COUNTRY_MAP = {
    "türk filmi": "TR",
    "türk yapımı": "TR",
    "yerli film": "TR",
    "kore filmi": "KR",
    "kore yapımı": "KR",
    "japon filmi": "JP",
    "fransız filmi": "FR",
    "amerikan filmi": "US",
    "hollywood": "US",
    "ingiliz filmi": "GB",
    "ingilizce bir": "GB",
    "çin filmi": "CN",
    "hint filmi": "IN",
    "bollywood": "IN",
    "fransız gerilimi": "FR",
}

COUNTRY_TO_LANGUAGE = {
    "TR": "tr",
    "KR": "ko",
    "JP": "ja",
    "FR": "fr",
    "US": "en",
    "GB": "en",
    "CN": "zh",
    "IN": "hi",
}

RATING_PATTERNS = {
    "high": ["ödüllü", "imdb puanı yüksek", "çok iyi puanlı", "oscar", "altın küre", "cannes"],
    "popular": ["çok izlenen", "popüler", "gişe rekoru", "blockbuster"],
}

NEGATION_SUFFIXES = [
    " olmasın",
    " istemiyorum",
    " hariç",
    " dışında",
    " sevmiyorum",
]

FALLBACK_BLOCKERS = [
    "olmasın",
    "istemiyorum",
    "hariç",
    "dışında",
    "sevmiyorum",
    "olmayacak",
]

def normalize_text(text: str) -> str:
    text = text.lower()

    replacements = {
        "’": "'",
        "“": '"',
        "”": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    decade_fixes = {
        "60'larda": "60larda",
        "70'lerde": "70lerde",
        "80'lerde": "80lerde",
        "90'larda": "90larda",
        "2000'lerde": "2000lerde",
        "2010'larda": "2010larda",
        "2020'lerde": "2020lerde",
    }
    for old, new in decade_fixes.items():
        text = text.replace(old, new)

    return text


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def detect_rating_preference(text_norm: str) -> str | None:
    for pref, patterns in RATING_PATTERNS.items():
        if contains_any(text_norm, patterns):
            return pref
    return None


def find_excluded_moods(text_norm: str) -> set[str]:
    excluded = set()
    for mood_value, patterns in NEGATIVE_MOOD_PATTERNS.items():
        if contains_any(text_norm, patterns):
            excluded.add(mood_value)
    return excluded


def find_excluded_genres(text_norm: str) -> set[int]:
    excluded = set()

    # düz eşleşmeler
    for genre_phrase, genre_id in GENRE_MAP.items():
        for suffix in NEGATION_SUFFIXES:
            if f"{genre_phrase}{suffix}" in text_norm:
                excluded.add(genre_id)
                break

    for synonym, base_genre in GENRE_SYNONYMS.items():
        genre_id = GENRE_MAP.get(base_genre)
        if genre_id is None:
            continue
        for suffix in NEGATION_SUFFIXES:
            if f"{synonym}{suffix}" in text_norm:
                excluded.add(genre_id)
                break

    # özel varyantlar
    if "romantizm olmasın" in text_norm:
        excluded.add(10749)

    # çoklu hariç yapıları
    if "aksiyon ve gerilim hariç" in text_norm or "gerilim ve aksiyon hariç" in text_norm:
        excluded.add(28)
        excluded.add(53)

    return excluded


def should_use_sentiment_fallback(text_norm: str) -> bool:
    if contains_any(text_norm, FALLBACK_BLOCKERS):
        return False

    all_mood_terms = list(MOOD_KEYWORDS.keys())
    all_strong_mood_patterns = [
        pattern
        for patterns in STRONG_MOOD_PATTERNS.values()
        for pattern in patterns
    ]

    if contains_any(text_norm, all_mood_terms + all_strong_mood_patterns):
        return False

    # Bu tip yalın filtre cümlelerinde mood uydurma
    neutral_request_patterns = [
        "öner",
        "film",
        "arıyorum",
        "istiyorum",
        "çıkan",
        "geçen",
    ]
    if contains_any(text_norm, neutral_request_patterns):
        return False

    if len(text_norm.split()) <= 2:
        return False

    return True


def parse(text: str) -> dict:
    text_norm = normalize_text(text)
    doc = nlp(text_norm)

    excluded_genres = find_excluded_genres(text_norm)
    excluded_moods = find_excluded_moods(text_norm)

    genre_ids_set = set()
    theme_set = set()

    params = {
        "genre_ids": [],
        "exclude_genre_ids": [],
        "year_gte": None,
        "year_lte": None,
        "mood": None,
        "excluded_moods": [],
        "theme": [],
        "low_violence": False,
        "high_violence": False,
        "runtime_pref": None,
        "original_language": None,
        "country": None,
        "rating_pref": None,
    }

    # Genre
    for genre_phrase, genre_id in GENRE_MAP.items():
        if genre_phrase in text_norm and genre_id not in excluded_genres:
            genre_ids_set.add(genre_id)

    for synonym, base_genre in GENRE_SYNONYMS.items():
        if synonym in text_norm:
            genre_id = GENRE_MAP[base_genre]
            if genre_id not in excluded_genres:
                genre_ids_set.add(genre_id)

    # Decade
    for decade_phrase, years in DECADE_MAP.items():
        if decade_phrase in text_norm:
            params["year_gte"] = years["year_gte"]
            params["year_lte"] = years["year_lte"]
            break

    # Theme
    for theme_phrase, theme_value in THEME_MAP.items():
        if theme_phrase in text_norm:
            theme_set.add(theme_value)

    # Violence
    if contains_any(text_norm, LOW_VIOLENCE_PATTERNS):
        params["low_violence"] = True

    # low_violence varsa high açma
    if not params["low_violence"] and contains_any(text_norm, HIGH_VIOLENCE_PATTERNS):
        params["high_violence"] = True

    # Runtime
    for runtime_key, runtime_patterns in RUNTIME_PATTERNS.items():
        if contains_any(text_norm, runtime_patterns):
            params["runtime_pref"] = runtime_key
            break

    # Language
    for lang_word, lang_code in LANGUAGE_MAP.items():
        if lang_word in text_norm:
            params["original_language"] = lang_code
            break

    # Country
    for country_phrase, country_code in COUNTRY_MAP.items():
        if country_phrase in text_norm:
            params["country"] = country_code
            break

    if params["country"] and params["original_language"] is None:
        params["original_language"] = COUNTRY_TO_LANGUAGE.get(params["country"])

    # Rating preference
    params["rating_pref"] = detect_rating_preference(text_norm)

    # Mood - strong patterns first
    for mood_value, patterns in STRONG_MOOD_PATTERNS.items():
        if mood_value in excluded_moods:
            continue
        if contains_any(text_norm, patterns):
            params["mood"] = mood_value
            break

    # Mood - keyword fallback
    if params["mood"] is None:
        for mood_word, mood_value in MOOD_KEYWORDS.items():
            if mood_value in excluded_moods:
                continue
            if mood_word in text_norm:
                params["mood"] = mood_value
                break

    # Token-based extra scan
    for token in doc:
        word = token.text.strip()

        if word in GENRE_MAP and GENRE_MAP[word] not in excluded_genres:
            genre_ids_set.add(GENRE_MAP[word])

        if word in GENRE_SYNONYMS:
            base_genre = GENRE_SYNONYMS[word]
            genre_id = GENRE_MAP[base_genre]
            if genre_id not in excluded_genres:
                genre_ids_set.add(genre_id)

        if word in THEME_MAP:
            theme_set.add(THEME_MAP[word])

    # Sentiment fallback
    if (
        params["mood"] is None
        and len(excluded_moods) == 0
        and should_use_sentiment_fallback(text_norm)
    ):
        sentiment_result = sentiment_analyzer(text)[0]
        if sentiment_result["label"] == "negative":
            params["mood"] = "dark"
        elif sentiment_result["label"] == "positive":
            params["mood"] = "light"

    params["genre_ids"] = sorted(list(genre_ids_set))
    params["exclude_genre_ids"] = sorted(list(excluded_genres))
    params["excluded_moods"] = sorted(list(excluded_moods))
    params["theme"] = sorted(list(theme_set))

    return params


if __name__ == "__main__":
    test_sentences = [
        "90'larda geçen, az kanlı bir korku filmi",
        "İçimi karartacak çok karanlık bir dram arıyorum",
        "Bilim kurgu ama çok gergin olmayan bir film",
        "Seri katilli twistli gerilim filmi öner",
        "2000'lerde geçen aksiyon filmi",
        "Eğlenceli bir komedi filmi istiyorum",
        "Uzayda geçen bilim kurgu öner",
        "Az şiddetli bir gerilim filmi arıyorum",
        "Zombi temalı korku filmi öner",
        "2010'larda geçen karanlık bir dram",
        "Korku olmasın, gerilim olabilir",
        "Dram istemiyorum ama romantik komedi olsun",
        "Aksiyon hariç bilim kurgu öner",
        "Komedi dışında duygusal bir film arıyorum",
        "Kısa bir kore filmi istiyorum",
        "Uzun ve ingilizce bir suç filmi öner",
        "Eski bir klasik film öner",
        "Son yıllardan eğlenceli bir aile filmi",
        "Sert olmayan korkutucu bir film istiyorum",
        "Romantik olmasın ama duygusal olsun",
        "Oscar ödüllü bir dram öner",
        "Bollywood'dan eğlenceli bir romantik film",
        "Hayalet temalı doğaüstü bir korku filmi",
        "Psikolojik gerilim ama çok karanlık olmasın",
        "İntikam temalı Fransız filmi öner",
        "80'lerde geçen, uzay temalı bir bilim kurgu arıyorum.",
        "Bugün çok yorgunum, kafamı dağıtacak çerezlik bir şeyler izlemek istiyorum.",
        "Kan gövdeyi götürmesin, az kanlı ama beni gerecek bir gerilim filmi olsun.",
        "İçinde seri katil olan sürpriz sonlu karanlık bir korku filmi önerebilir misin?",
        "2000'lerde çekilmiş çok eğlenceli bir komedi filmi arıyorum, kahkaha atmak istiyorum.",
        "Zombilerin olduğu, hayatta kalma mücadelesi veren üzücü bir dram filmi var mı?",
        "Hayattan çok soğudum, içimi ısıtacak tatlı bir animasyon izlemek istiyorum.",
        "70'lerden kalma eski ama çok şiddetli olmayan bir aksiyon ve macera filmi.",
        "Kız arkadaşımla izleyeceğiz, hem biraz romantik olsun hem de uzayda geçsin.",
        "2010'larda çıkan twistli aksiyon.",
        "Çok yorgunum, kafamı tamamen dağıtacak çerezlik bir komedi filmi açayım.",
        "Hayattan soğudum, içimi ısıtacak pozitif bir animasyon izlemek istiyorum.",
        "Çok kanlı olmasın ama beni acayip gerecek bir gerilim filmi arıyorum.",
        "Kore yapımı kısa bir dram arıyorum ama içinde romantizm olmasın.",
        "Oscar ödüllü bir bilim kurgu istiyorum ama sakın korkutucu olmasın.",
        "Aksiyon ve gerilim hariç, sadece 2020'lerde çıkmış Bollywood yapımı romantik film.",
        "60'lardan kalma çok uzun bir suç filmi.",
        "Uzayda geçen hayatta kalma temalı İngilizce bir bilim kurgu.",
        "Beni koltuğuma çivileyecek, nefes kesen Fransız gerilimi.",
        "90'lardan bir dram arıyorum ama çok karanlık ve depresif olmasın.",
    ]

    print("=" * 60)
    print("TEST SONUÇLARI")
    print("=" * 60)

    for sentence in test_sentences:
        print(f"\nKullanıcının yazdığı: '{sentence}'")
        sonuc = parse(sentence)
        print(f"Bilgisayarın anladığı: {sonuc}")