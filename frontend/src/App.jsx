import { useEffect, useState } from "react";
import axios from "axios";
import SearchBar from "./components/SearchBar";
import MovieCard from "./components/MovieCard";

const API_BASE_URL = import.meta.env.VITE_API_URL;
const TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500";

const EXAMPLE_QUERIES = [
  "90'larda geçen karanlık bir gerilim filmi",
  "Az şiddetli ama sürükleyici bir korku filmi",
  "Twist sonlu bilim kurgu filmi",
];

function SkeletonCard() {
  return (
    <div style={styles.skeletonCard}>
      <div style={styles.skeletonImage} />
      <div style={styles.skeletonContent}>
        <div style={styles.skeletonLineWide} />
        <div style={styles.skeletonLineShort} />
        <div style={styles.skeletonLineWide} />
        <div style={styles.skeletonLineMedium} />
        <div style={styles.skeletonBox} />
        <div style={styles.skeletonBars} />
        <div style={styles.skeletonPalette} />
      </div>
    </div>
  );
}

function App() {
  const [healthStatus, setHealthStatus] = useState("Kontrol ediliyor...");
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [noResults, setNoResults] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [showWarmupMessage, setShowWarmupMessage] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      if (!API_BASE_URL) {
        setHealthStatus("API adresi eksik");
        return;
      }

      try {
        const response = await axios.get(`${API_BASE_URL}/health`);
        setHealthStatus(response?.data?.status || "ok");
      } catch {
        setHealthStatus("Backend'e bağlanılamadı");
      }
    };

    checkHealth();
  }, []);

  const handleSearch = async (query) => {
    const trimmedQuery = query.trim();

    if (!trimmedQuery) return;

    if (!API_BASE_URL) {
      setHasSearched(true);
      setMovies([]);
      setNoResults(false);
      setErrorMessage("API adresi tanımlı değil. VITE_API_URL değerini kontrol et.");
      return;
    }

    setLoading(true);
    setErrorMessage("");
    setNoResults(false);
    setHasSearched(true);
    setShowWarmupMessage(false);
    setMovies([]);

    const warmupTimer = setTimeout(() => {
      setShowWarmupMessage(true);
    }, 1500);

    try {
      const response = await axios.get(`${API_BASE_URL}/search`, {
        params: { q: trimmedQuery },
      });

      const results = Array.isArray(response?.data?.results)
        ? response.data.results
        : [];

      const formattedMovies = results.map((movie, index) => ({
        ...movie,
        tmdb_id: movie.tmdb_id ?? movie.id ?? `movie-${index}`,
        poster_url: movie.poster_url
          ? movie.poster_url
          : movie.poster_path
            ? `${TMDB_IMAGE_BASE_URL}${movie.poster_path}`
            : "",
        year:
          movie.year ??
          (movie.release_date ? String(movie.release_date).slice(0, 4) : ""),
      }));

      setMovies(formattedMovies);

      if (formattedMovies.length === 0) {
        setNoResults(true);
      }
    } catch (error) {
      console.error("Search error:", error);
      setMovies([]);
      setNoResults(false);
      setErrorMessage("Arama sırasında bir hata oluştu. Lütfen tekrar dene.");
    } finally {
      clearTimeout(warmupTimer);
      setLoading(false);
      setShowWarmupMessage(false);
    }
  };

  const healthOk = ["ok", "healthy", "up"].includes(
    String(healthStatus).toLowerCase()
  );

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <header style={styles.hero}>
          <div style={styles.badgeRow}>
            <span style={styles.badge}>Akıllı Film Öneri Sistemi</span>
            <span
              style={{
                ...styles.statusBadge,
                backgroundColor: healthOk
                  ? "rgba(22, 163, 74, 0.18)"
                  : "rgba(220, 38, 38, 0.18)",
                color: healthOk ? "#86efac" : "#fca5a5",
              }}
            >
              API: {healthStatus}
            </span>
          </div>

          <img
            src="/cue-hero.jpeg"
            alt="Cue visual"
            style={styles.heroBanner}
          />

          <h1 style={styles.title}>
            <span style={styles.titleCue}>Cue</span>
            <span style={styles.titleRest}>Smart Movie Recommendation</span>
          </h1>

          <p style={styles.subtitle}>
            Doğal dilde yazdığın isteği anlayıp, duygu eğrisi ve görsel ipuçlarıyla
            zenginleştirilmiş film önerileri sunan akıllı keşif arayüzü.
          </p>

          <section style={styles.searchCard}>
            <SearchBar onSearch={handleSearch} loading={loading} />

            {!hasSearched && !loading && (
              <>
                <p style={styles.searchHint}>
                  Örnek bir istek yaz veya aşağıdaki aramalardan birini dene.
                </p>

                <div style={styles.exampleList}>
                  {EXAMPLE_QUERIES.map((item) => (
                    <button
                      key={item}
                      type="button"
                      style={styles.exampleChip}
                      onClick={() => handleSearch(item)}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </>
            )}
          </section>
        </header>

        {!loading && errorMessage && (
          <div style={styles.errorBox}>{errorMessage}</div>
        )}

        {!loading && noResults && !errorMessage && (
          <div style={styles.infoBox}>
            Aramana uygun sonuç bulunamadı. Daha farklı bir ifade veya daha genel
            bir açıklama deneyebilirsin.
          </div>
        )}

        {loading && showWarmupMessage && (
          <div style={styles.warmupBox}>
            Sunucu hazırlanıyor, ilk istek birkaç saniye sürebilir.
          </div>
        )}

        {(loading || movies.length > 0) && (
          <section style={styles.resultsSection}>
            <div style={styles.resultsHeader}>
              <h2 style={styles.resultsTitle}>
                {loading ? "Öneriler hazırlanıyor..." : "Önerilen Filmler"}
              </h2>
            </div>

            <div style={styles.grid}>
              {loading
                ? Array.from({ length: 3 }).map((_, index) => (
                    <SkeletonCard key={index} />
                  ))
                : movies.map((movie) => (
                    <MovieCard key={movie.tmdb_id} movie={movie} />
                  ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

const shimmer = {
  background:
    "linear-gradient(90deg, rgba(51,65,85,0.55) 25%, rgba(100,116,139,0.38) 50%, rgba(51,65,85,0.55) 75%)",
  backgroundSize: "200% 100%",
  animation: "shimmer 1.4s infinite linear",
};

const styles = {
  page: {
    minHeight: "100vh",
    background: `
      radial-gradient(circle at 18% 20%, rgba(139, 92, 246, 0.28), transparent 26%),
      radial-gradient(circle at 82% 10%, rgba(59, 130, 246, 0.12), transparent 22%),
      radial-gradient(circle at 50% 100%, rgba(168, 85, 247, 0.10), transparent 28%),
      linear-gradient(180deg, #080d1d 0%, #0a1023 48%, #090f20 100%)
    `,
    color: "white",
    padding: "28px 20px 56px",
  },
  container: {
    maxWidth: "1200px",
    margin: "0 auto",
  },
  hero: {
    textAlign: "center",
    paddingTop: "4px",
    marginBottom: "28px",
  },
  badgeRow: {
    display: "flex",
    justifyContent: "center",
    gap: "10px",
    flexWrap: "wrap",
    marginBottom: "18px",
  },
  badge: {
    padding: "8px 14px",
    borderRadius: "999px",
    background: "rgba(99, 102, 241, 0.16)",
    color: "#c7d2fe",
    fontSize: "13px",
    fontWeight: "600",
    border: "1px solid rgba(129, 140, 248, 0.22)",
    boxShadow: "0 0 24px rgba(99, 102, 241, 0.08)",
  },
  statusBadge: {
    padding: "8px 14px",
    borderRadius: "999px",
    fontSize: "13px",
    fontWeight: "600",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  heroBanner: {
    width: "100%",
    maxWidth: "760px",
    height: "auto",
    display: "block",
    margin: "8px auto 26px auto",
    borderRadius: "28px",
    objectFit: "contain",
    boxShadow:
      "0 0 38px rgba(168, 85, 247, 0.20), 0 0 72px rgba(59, 130, 246, 0.10)",
  },
  title: {
    margin: "0 0 14px 0",
  },
  titleCue: {
    display: "block",
    fontSize: "clamp(68px, 11vw, 126px)",
    lineHeight: "0.9",
    fontWeight: "800",
    letterSpacing: "-0.08em",
    background: "linear-gradient(90deg, #ffffff 0%, #eadcff 42%, #b5c8ff 100%)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    textShadow: "0 0 36px rgba(168, 85, 247, 0.20)",
  },
  titleRest: {
    display: "block",
    marginTop: "6px",
    fontSize: "clamp(22px, 3.2vw, 44px)",
    lineHeight: "1.08",
    fontWeight: "700",
    color: "#f1f5f9",
    letterSpacing: "-0.03em",
  },
  subtitle: {
    maxWidth: "760px",
    margin: "0 auto 28px auto",
    color: "#cbd5e1",
    fontSize: "15px",
    lineHeight: "1.8",
  },
  searchCard: {
    maxWidth: "920px",
    margin: "0 auto",
    padding: "28px 26px",
    borderRadius: "28px",
    background:
      "linear-gradient(180deg, rgba(12,18,40,0.82) 0%, rgba(10,15,35,0.76) 100%)",
    border: "1px solid rgba(168, 85, 247, 0.14)",
    backdropFilter: "blur(12px)",
    boxShadow:
      "0 24px 70px rgba(0,0,0,0.24), inset 0 1px 0 rgba(255,255,255,0.03)",
  },
  searchHint: {
    margin: "18px 0 0 0",
    color: "#cbd5e1",
    fontSize: "14px",
    lineHeight: "1.7",
  },
  exampleList: {
    marginTop: "18px",
    display: "flex",
    justifyContent: "center",
    flexWrap: "wrap",
    gap: "12px",
  },
  exampleChip: {
    padding: "12px 18px",
    borderRadius: "999px",
    border: "1px solid rgba(129, 140, 248, 0.24)",
    background:
      "linear-gradient(180deg, rgba(30,41,59,0.95) 0%, rgba(17,24,39,0.95) 100%)",
    color: "#e0e7ff",
    cursor: "pointer",
    fontSize: "13px",
    fontWeight: "600",
    boxShadow: "0 0 20px rgba(99, 102, 241, 0.06)",
  },
  infoBox: {
    textAlign: "center",
    color: "#d1d5db",
    margin: "0 auto 20px auto",
    maxWidth: "920px",
    padding: "14px 16px",
    borderRadius: "16px",
    background: "rgba(31, 41, 55, 0.9)",
    border: "1px solid rgba(255,255,255,0.06)",
  },
  errorBox: {
    textAlign: "center",
    color: "#fecaca",
    fontWeight: "600",
    margin: "0 auto 20px auto",
    maxWidth: "920px",
    padding: "14px 16px",
    borderRadius: "16px",
    background: "rgba(127, 29, 29, 0.35)",
    border: "1px solid rgba(248, 113, 113, 0.25)",
  },
  warmupBox: {
    textAlign: "center",
    color: "#fde68a",
    fontWeight: "600",
    margin: "0 auto 20px auto",
    maxWidth: "920px",
    padding: "14px 16px",
    borderRadius: "16px",
    background: "rgba(120, 53, 15, 0.35)",
    border: "1px solid rgba(251, 191, 36, 0.25)",
  },
  resultsSection: {
    marginTop: "26px",
  },
  resultsHeader: {
    marginBottom: "16px",
  },
  resultsTitle: {
    margin: 0,
    fontSize: "28px",
    color: "#f8fafc",
    textAlign: "center",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
    gap: "20px",
    alignItems: "start",
  },
  skeletonCard: {
    background:
      "linear-gradient(180deg, rgba(31,41,55,1) 0%, rgba(17,24,39,1) 100%)",
    borderRadius: "20px",
    overflow: "hidden",
    boxShadow: "0 12px 30px rgba(0,0,0,0.28)",
    border: "1px solid rgba(255,255,255,0.06)",
  },
  skeletonImage: {
    height: "260px",
    ...shimmer,
  },
  skeletonContent: {
    padding: "18px",
  },
  skeletonLineWide: {
    height: "18px",
    borderRadius: "10px",
    marginBottom: "12px",
    width: "85%",
    ...shimmer,
  },
  skeletonLineShort: {
    height: "14px",
    borderRadius: "10px",
    marginBottom: "12px",
    width: "35%",
    ...shimmer,
  },
  skeletonLineMedium: {
    height: "14px",
    borderRadius: "10px",
    marginBottom: "14px",
    width: "70%",
    ...shimmer,
  },
  skeletonBox: {
    height: "70px",
    borderRadius: "14px",
    marginBottom: "14px",
    ...shimmer,
  },
  skeletonBars: {
    height: "110px",
    borderRadius: "14px",
    marginBottom: "14px",
    ...shimmer,
  },
  skeletonPalette: {
    height: "34px",
    borderRadius: "999px",
    width: "40%",
    ...shimmer,
  },
};

export default App;
