import { useEffect, useState } from "react";
import axios from "axios";
import SearchBar from "../components/SearchBar";
import MovieCard from "../components/MovieCard";

const API_BASE_URL = import.meta.env.VITE_API_URL;
const TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500";
const SEARCH_STORAGE_KEY = "cue_last_search_state";

const EXAMPLE_QUERIES = [
  "90'larda geçen karanlık bir gerilim filmi",
  "Az şiddetli ama sürükleyici bir korku filmi",
  "Twist sonlu bilim kurgu filmi",
];

function SkeletonCard() {
  return (
    <div style={styles.skeletonCard}>
      <div style={styles.skeletonPoster} />
      <div style={styles.skeletonBody}>
        <div style={styles.skeletonTitle} />
        <div style={styles.skeletonMeta} />
        <div style={styles.skeletonTextLong} />
        <div style={styles.skeletonTextMedium} />
      </div>
    </div>
  );
}

function SearchPage() {
  const [healthStatus, setHealthStatus] = useState("Kontrol ediliyor...");
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [noResults, setNoResults] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [showWarmupMessage, setShowWarmupMessage] = useState(false);
  const [lastQuery, setLastQuery] = useState("");
  const [removedMovies, setRemovedMovies] = useState([]);

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

    const savedState = sessionStorage.getItem(SEARCH_STORAGE_KEY);

    if (savedState) {
      try {
        const parsed = JSON.parse(savedState);

        setMovies(Array.isArray(parsed.movies) ? parsed.movies : []);
        setHasSearched(Boolean(parsed.hasSearched));
        setNoResults(Boolean(parsed.noResults));
        setLastQuery(parsed.lastQuery || "");
        setRemovedMovies(
          Array.isArray(parsed.removedMovies) ? parsed.removedMovies : []
        );
      } catch (error) {
        console.error("Saved search state parse error:", error);
      }
    }
  }, []);

  const persistSearchState = ({
    moviesValue,
    hasSearchedValue,
    noResultsValue,
    lastQueryValue,
    removedMoviesValue,
  }) => {
    sessionStorage.setItem(
      SEARCH_STORAGE_KEY,
      JSON.stringify({
        movies: moviesValue,
        hasSearched: hasSearchedValue,
        noResults: noResultsValue,
        lastQuery: lastQueryValue,
        removedMovies: removedMoviesValue,
      })
    );
  };

  const handleSearch = async (query) => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    setLastQuery(trimmedQuery);
    setRemovedMovies([]);

    if (!API_BASE_URL) {
      setHasSearched(true);
      setMovies([]);
      setNoResults(false);
      setErrorMessage(
        "API adresi tanımlı değil. VITE_API_URL değerini kontrol et."
      );

      persistSearchState({
        moviesValue: [],
        hasSearchedValue: true,
        noResultsValue: false,
        lastQueryValue: trimmedQuery,
        removedMoviesValue: [],
      });
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

      const isNoResults = formattedMovies.length === 0;

      setMovies(formattedMovies);
      setNoResults(isNoResults);
      setRemovedMovies([]);

      persistSearchState({
        moviesValue: formattedMovies,
        hasSearchedValue: true,
        noResultsValue: isNoResults,
        lastQueryValue: trimmedQuery,
        removedMoviesValue: [],
      });
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

  const handleRemoveMovie = (movieToRemove) => {
    const updatedMovies = movies.filter(
      (movie) => movie.tmdb_id !== movieToRemove.tmdb_id
    );
    const updatedRemovedMovies = [movieToRemove, ...removedMovies];

    setMovies(updatedMovies);
    setRemovedMovies(updatedRemovedMovies);
    setNoResults(updatedMovies.length === 0);

    persistSearchState({
      moviesValue: updatedMovies,
      hasSearchedValue: hasSearched,
      noResultsValue: updatedMovies.length === 0,
      lastQueryValue: lastQuery,
      removedMoviesValue: updatedRemovedMovies,
    });
  };

  const handleRestoreLastRemoved = () => {
    if (removedMovies.length === 0) return;

    const [lastRemoved, ...remainingRemoved] = removedMovies;
    const restoredMovies = [lastRemoved, ...movies];

    setMovies(restoredMovies);
    setRemovedMovies(remainingRemoved);
    setNoResults(false);

    persistSearchState({
      moviesValue: restoredMovies,
      hasSearchedValue: hasSearched,
      noResultsValue: false,
      lastQueryValue: lastQuery,
      removedMoviesValue: remainingRemoved,
    });
  };

  const healthOk = ["ok", "healthy", "up"].includes(
    String(healthStatus).toLowerCase()
  );

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <section style={styles.heroCard}>
          <div style={styles.topRow}>
            <div style={styles.titleArea}>
              <h1 style={styles.pageTitle}>Film Arama</h1>
              <p style={styles.pageSubtitle}>
                Aradığın tarza en yakın film önerilerini keşfet.
              </p>
            </div>

            <div style={styles.statusArea}>
              <span style={styles.badge}>Akıllı Film Öneri Sistemi</span>
              <span
                style={{
                  ...styles.statusBadge,
                  background: healthOk
                    ? "rgba(22,163,74,0.15)"
                    : "rgba(220,38,38,0.15)",
                  color: healthOk ? "#86efac" : "#fca5a5",
                  border: healthOk
                    ? "1px solid rgba(34,197,94,0.24)"
                    : "1px solid rgba(248,113,113,0.24)",
                }}
              >
                API: {healthStatus}
              </span>
            </div>
          </div>

          <div style={styles.searchPanel}>
            <SearchBar onSearch={handleSearch} loading={loading} />
          </div>

          {!hasSearched && !loading && (
            <div style={styles.exampleWrap}>
              <p style={styles.exampleText}>
                Örnek aramalarla hızlıca başlayabilirsin:
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
            </div>
          )}
        </section>

        {!loading && errorMessage && (
          <div style={styles.errorBox}>{errorMessage}</div>
        )}

        {!loading && noResults && !errorMessage && (
          <div style={styles.infoBox}>
            Aramana uygun sonuç bulunamadı. Daha genel ya da farklı bir ifade
            deneyebilirsin.
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
              <div style={styles.resultsHeaderLeft}>
                <h2 style={styles.resultsTitle}>Önerilen Filmler</h2>
                {lastQuery && !loading && (
                  <p style={styles.resultsSubtitle}>
                    Son sorgu:{" "}
                    <span style={styles.resultsQuery}>{lastQuery}</span>
                  </p>
                )}
              </div>

              {removedMovies.length > 0 && !loading && (
                <div style={styles.undoBox}>
                  <span style={styles.undoText}>
                    {removedMovies[0]?.title} kaldırıldı
                  </span>
                  <button
                    type="button"
                    style={styles.undoButton}
                    onClick={handleRestoreLastRemoved}
                  >
                    Geri Getir
                  </button>
                </div>
              )}
            </div>

            <div style={styles.grid}>
              {loading
                ? Array.from({ length: 3 }).map((_, index) => (
                    <SkeletonCard key={index} />
                  ))
                : movies.map((movie) => (
                    <MovieCard
                      key={movie.tmdb_id}
                      movie={movie}
                      onRemove={handleRemoveMovie}
                    />
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
    "linear-gradient(90deg, rgba(42,55,88,0.45) 25%, rgba(88,104,141,0.24) 50%, rgba(42,55,88,0.45) 75%)",
  backgroundSize: "200% 100%",
  animation: "shimmer 1.4s infinite linear",
};

const styles = {
  page: {
    minHeight: "100vh",
    width: "100%",
    boxSizing: "border-box",
    background: `
      radial-gradient(circle at 18% 16%, rgba(99, 102, 241, 0.14), transparent 24%),
      radial-gradient(circle at 82% 12%, rgba(59, 130, 246, 0.10), transparent 22%),
      radial-gradient(circle at 50% 100%, rgba(139, 92, 246, 0.08), transparent 26%),
      linear-gradient(180deg, #060b18 0%, #081224 52%, #050a16 100%)
    `,
    color: "#f8fafc",
    padding: "22px 16px 56px",
  },

  container: {
    width: "100%",
    maxWidth: "100%",
    boxSizing: "border-box",
  },

  heroCard: {
    width: "100%",
    boxSizing: "border-box",
    padding: "28px 28px 24px",
    borderRadius: "24px",
    background:
      "linear-gradient(180deg, rgba(10,16,32,0.96) 0%, rgba(7,12,26,0.93) 100%)",
    border: "1px solid rgba(148,163,184,0.14)",
    boxShadow: "0 18px 42px rgba(0,0,0,0.18)",
    marginBottom: "26px",
  },

  topRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: "20px",
    flexWrap: "wrap",
    marginBottom: "22px",
  },

  titleArea: {
    minWidth: 0,
    flex: "1 1 560px",
  },

  statusArea: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
    alignItems: "center",
  },

  pageTitle: {
    margin: "0 0 10px",
    fontSize: "44px",
    fontWeight: "800",
    lineHeight: 1.05,
    letterSpacing: "-0.03em",
    color: "#ffffff",
  },

  pageSubtitle: {
    margin: 0,
    fontSize: "18px",
    lineHeight: "1.7",
    color: "#cbd5e1",
    maxWidth: "760px",
  },

  badge: {
    padding: "10px 15px",
    borderRadius: "999px",
    border: "1px solid rgba(129,140,248,0.20)",
    background: "rgba(99,102,241,0.12)",
    color: "#c7d2fe",
    fontSize: "13px",
    fontWeight: "700",
    whiteSpace: "nowrap",
  },

  statusBadge: {
    padding: "10px 15px",
    borderRadius: "999px",
    fontSize: "13px",
    fontWeight: "700",
    whiteSpace: "nowrap",
  },

  searchPanel: {
    width: "100%",
    boxSizing: "border-box",
    padding: "18px",
    borderRadius: "20px",
    background: "rgba(8, 13, 28, 0.92)",
    border: "1px solid rgba(148,163,184,0.12)",
    boxShadow: "inset 0 1px 0 rgba(255,255,255,0.03)",
  },

  exampleWrap: {
    marginTop: "18px",
  },

  exampleText: {
    margin: "0 0 12px",
    color: "#cbd5e1",
    fontSize: "15px",
    lineHeight: "1.6",
  },

  exampleList: {
    display: "flex",
    flexWrap: "wrap",
    gap: "10px",
  },

  exampleChip: {
    padding: "11px 16px",
    borderRadius: "999px",
    border: "1px solid rgba(129,140,248,0.18)",
    background: "rgba(20, 28, 48, 0.95)",
    color: "#e2e8f0",
    cursor: "pointer",
    fontSize: "14px",
    fontWeight: "600",
    lineHeight: "1.4",
  },

  errorBox: {
    marginBottom: "18px",
    padding: "18px 20px",
    borderRadius: "18px",
    background: "rgba(127,29,29,0.20)",
    border: "1px solid rgba(248,113,113,0.22)",
    color: "#fecaca",
    fontSize: "16px",
    lineHeight: "1.65",
    fontWeight: "600",
  },

  infoBox: {
    marginBottom: "18px",
    padding: "18px 20px",
    borderRadius: "18px",
    background: "rgba(15,23,42,0.85)",
    border: "1px solid rgba(148,163,184,0.12)",
    color: "#dbeafe",
    fontSize: "16px",
    lineHeight: "1.65",
  },

  warmupBox: {
    marginBottom: "18px",
    padding: "18px 20px",
    borderRadius: "18px",
    background: "rgba(120,53,15,0.20)",
    border: "1px solid rgba(251,191,36,0.22)",
    color: "#fde68a",
    fontSize: "16px",
    lineHeight: "1.65",
    fontWeight: "600",
  },

  resultsSection: {
    width: "100%",
    boxSizing: "border-box",
    padding: "26px 24px 28px",
    borderRadius: "24px",
    background:
      "linear-gradient(180deg, rgba(8,13,28,0.80) 0%, rgba(7,12,26,0.62) 100%)",
    border: "1px solid rgba(148,163,184,0.10)",
    boxShadow: "0 14px 36px rgba(0,0,0,0.12)",
  },

  resultsHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "18px",
    flexWrap: "wrap",
    marginBottom: "22px",
    paddingBottom: "16px",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
  },

  resultsHeaderLeft: {
    minWidth: 0,
  },

  resultsTitle: {
    margin: 0,
    fontSize: "36px",
    fontWeight: "800",
    lineHeight: 1.1,
    letterSpacing: "-0.02em",
    color: "#ffffff",
  },

  resultsSubtitle: {
    margin: "10px 0 0",
    fontSize: "16px",
    color: "#cbd5e1",
    lineHeight: "1.6",
  },

  resultsQuery: {
    color: "#ffffff",
    fontWeight: "700",
  },

  undoBox: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    flexWrap: "wrap",
    padding: "12px 14px",
    borderRadius: "14px",
    background: "rgba(9, 18, 36, 0.92)",
    border: "1px solid rgba(96,165,250,0.18)",
  },

  undoText: {
    color: "#dbeafe",
    fontSize: "15px",
    fontWeight: "700",
    lineHeight: "1.5",
  },

  undoButton: {
    padding: "11px 15px",
    borderRadius: "10px",
    border: "1px solid rgba(96,165,250,0.22)",
    background: "rgba(26, 40, 68, 0.95)",
    color: "#bfdbfe",
    fontSize: "15px",
    fontWeight: "700",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },

  grid: {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
    width: "100%",
  },

  skeletonCard: {
    display: "grid",
    gridTemplateColumns: "180px 1fr",
    width: "100%",
    minHeight: "220px",
    borderRadius: "20px",
    overflow: "hidden",
    background:
      "linear-gradient(90deg, rgba(17,24,39,0.95) 0%, rgba(8,15,32,0.95) 100%)",
    border: "1px solid rgba(255,255,255,0.08)",
    boxShadow: "0 10px 28px rgba(0,0,0,0.18)",
  },

  skeletonPoster: {
    minHeight: "220px",
    ...shimmer,
  },

  skeletonBody: {
    padding: "26px 28px",
  },

  skeletonTitle: {
    height: "28px",
    width: "34%",
    borderRadius: "10px",
    marginBottom: "16px",
    ...shimmer,
  },

  skeletonMeta: {
    height: "18px",
    width: "18%",
    borderRadius: "10px",
    marginBottom: "20px",
    ...shimmer,
  },

  skeletonTextLong: {
    height: "18px",
    width: "84%",
    borderRadius: "10px",
    marginBottom: "12px",
    ...shimmer,
  },

  skeletonTextMedium: {
    height: "18px",
    width: "60%",
    borderRadius: "10px",
    ...shimmer,
  },
};

export default SearchPage;
