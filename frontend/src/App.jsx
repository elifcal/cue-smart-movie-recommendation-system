import { useEffect, useState } from "react";
import axios from "axios";
import SearchBar from "./components/SearchBar";
import MovieCard from "./components/MovieCard";

const API_BASE_URL = import.meta.env.VITE_API_URL;
const TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500";

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
  const [hasSearched, setHasSearched] = useState(false);
  const [showWarmupMessage, setShowWarmupMessage] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`);
        setHealthStatus(response.data.status || "ok");
      } catch {
        setHealthStatus("Backend'e bağlanılamadı");
      }
    };

    checkHealth();
  }, []);

  const handleSearch = async (query) => {
    if (!query.trim()) return;

    setLoading(true);
    setErrorMessage("");
    setHasSearched(true);
    setShowWarmupMessage(false);

    const warmupTimer = setTimeout(() => {
      setShowWarmupMessage(true);
    }, 1500);

    try {
      const response = await axios.get(`${API_BASE_URL}/search`, {
        params: { q: query },
      });

      const results = Array.isArray(response.data.results)
        ? response.data.results
        : [];

      const formattedMovies = results.map((movie) => ({
        ...movie,
        tmdb_id: movie.id,
        poster_url: movie.poster_path
          ? `${TMDB_IMAGE_BASE_URL}${movie.poster_path}`
          : "",
      }));

      setMovies(formattedMovies);

      if (formattedMovies.length === 0) {
        setErrorMessage("Sonuç bulunamadı.");
      }
    } catch (error) {
      console.error("Search error:", error);
      setMovies([]);
      setErrorMessage("Arama sırasında bir hata oluştu.");
    } finally {
      clearTimeout(warmupTimer);
      setLoading(false);
      setShowWarmupMessage(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <header style={styles.hero}>
          <div style={styles.badgeRow}>
            <span style={styles.badge}>Akıllı Film Öneri Sistemi</span>
            <span
              style={{
                ...styles.statusBadge,
                backgroundColor:
                  healthStatus === "ok"
                    ? "rgba(22, 163, 74, 0.18)"
                    : "rgba(220, 38, 38, 0.18)",
                color: healthStatus === "ok" ? "#86efac" : "#fca5a5",
              }}
            >
              API: {healthStatus}
            </span>
          </div>

          <h1 style={styles.title}>Cue Smart Movie Recommendation</h1>
          <p style={styles.subtitle}>
            Kullanıcının yazdığı isteğe göre film önerileri sunan, duygu eğrisi ve
            görsel palet gibi zengin açıklamalar gösteren öneri arayüzü.
          </p>
        </header>

        <section style={styles.searchSection}>
          <SearchBar onSearch={handleSearch} loading={loading} />
        </section>

        {!hasSearched && !loading && !errorMessage && (
          <div style={styles.infoBox}>
            Arama yaparak backend’den gerçek sonuçları getir.
          </div>
        )}

        {!loading && errorMessage && (
          <div style={styles.errorBox}>{errorMessage}</div>
        )}

        {loading && showWarmupMessage && (
          <div style={styles.warmupBox}>
            Sunucu hazırlanıyor, ilk istek birkaç saniye sürebilir.
          </div>
        )}

        <section style={styles.grid}>
          {loading
            ? Array.from({ length: 3 }).map((_, index) => (
                <SkeletonCard key={index} />
              ))
            : movies.map((movie) => (
                <MovieCard key={movie.tmdb_id} movie={movie} />
              ))}
        </section>
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
    background:
      "radial-gradient(circle at top, rgba(37,99,235,0.14), transparent 30%), #0f172a",
    color: "white",
    padding: "32px 20px",
  },
  container: {
    maxWidth: "1180px",
    margin: "0 auto",
  },
  hero: {
    textAlign: "center",
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
    background: "rgba(59, 130, 246, 0.18)",
    color: "#bfdbfe",
    fontSize: "13px",
    fontWeight: "600",
    border: "1px solid rgba(96, 165, 250, 0.25)",
  },
  statusBadge: {
    padding: "8px 14px",
    borderRadius: "999px",
    fontSize: "13px",
    fontWeight: "600",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  title: {
    fontSize: "clamp(28px, 5vw, 44px)",
    lineHeight: "1.1",
    marginBottom: "12px",
  },
  subtitle: {
    maxWidth: "760px",
    margin: "0 auto",
    color: "#cbd5e1",
    fontSize: "15px",
    lineHeight: "1.7",
  },
  searchSection: {
    background: "rgba(15, 23, 42, 0.7)",
    border: "1px solid rgba(148, 163, 184, 0.14)",
    borderRadius: "20px",
    padding: "20px",
    backdropFilter: "blur(10px)",
    marginBottom: "24px",
  },
  infoBox: {
    textAlign: "center",
    color: "#d1d5db",
    marginBottom: "20px",
    padding: "14px 16px",
    borderRadius: "14px",
    background: "rgba(31, 41, 55, 0.9)",
    border: "1px solid rgba(255,255,255,0.06)",
  },
  errorBox: {
    textAlign: "center",
    color: "#fecaca",
    fontWeight: "600",
    marginBottom: "20px",
    padding: "14px 16px",
    borderRadius: "14px",
    background: "rgba(127, 29, 29, 0.35)",
    border: "1px solid rgba(248, 113, 113, 0.25)",
  },
  warmupBox: {
    textAlign: "center",
    color: "#fde68a",
    fontWeight: "600",
    marginBottom: "20px",
    padding: "14px 16px",
    borderRadius: "14px",
    background: "rgba(120, 53, 15, 0.35)",
    border: "1px solid rgba(251, 191, 36, 0.25)",
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
