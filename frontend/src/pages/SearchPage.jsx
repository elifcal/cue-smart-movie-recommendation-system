import { useEffect, useLayoutEffect, useRef, useState } from "react";
import axios from "axios";
import SearchBar from "../components/SearchBar";
import MovieCard from "../components/MovieCard";

const API_BASE_URL = import.meta.env.VITE_API_URL;
const TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500";
const SEARCH_STORAGE_KEY = "cue_search_restore_once";
const PIN_TOP_OFFSET = 12;
const WARMUP_MESSAGE_DELAY = 10000;

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

function SearchLoadingState({ showWarmupMessage }) {
  return (
    <div style={styles.loadingStateWrap}>
      <div style={styles.loadingStateTop}>
        <div style={styles.loadingSpinner} />

        <div style={styles.loadingTextWrap}>
          <h3 style={styles.loadingTitle}>
            Cue sizin için en uygun filmleri arıyor
          </h3>
          <p style={styles.loadingSubtitle}>
            Bu işlem birkaç saniye sürebilir.
          </p>
        </div>
      </div>

      {showWarmupMessage && (
        <div style={styles.inlineInfoBox}>
          İlk istek biraz daha uzun sürebilir, sonuçlar hazırlanıyor.
        </div>
      )}

      <div style={styles.grid}>
        {Array.from({ length: 3 }).map((_, index) => (
          <SkeletonCard key={index} />
        ))}
      </div>
    </div>
  );
}

function SearchPage() {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [noResults, setNoResults] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [showWarmupMessage, setShowWarmupMessage] = useState(false);
  const [lastQuery, setLastQuery] = useState("");
  const [removedMovies, setRemovedMovies] = useState([]);
  const [isSearchPinned, setIsSearchPinned] = useState(false);
  const [searchBarHeight, setSearchBarHeight] = useState(0);

  const movieRefs = useRef({});
  const searchAnchorRef = useRef(null);
  const searchShellRef = useRef(null);

  useEffect(() => {
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
      } finally {
        sessionStorage.removeItem(SEARCH_STORAGE_KEY);
      }
    }
  }, []);

  useLayoutEffect(() => {
    const measure = () => {
      if (searchShellRef.current) {
        setSearchBarHeight(searchShellRef.current.offsetHeight);
      }
    };

    measure();
    window.addEventListener("resize", measure);

    return () => {
      window.removeEventListener("resize", measure);
    };
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      if (!searchAnchorRef.current) return;

      const anchorTop = searchAnchorRef.current.getBoundingClientRect().top;
      setIsSearchPinned(anchorTop <= PIN_TOP_OFFSET);
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleScroll);

    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
    };
  }, []);

  const persistStateForDetail = () => {
    sessionStorage.setItem(
      SEARCH_STORAGE_KEY,
      JSON.stringify({
        movies,
        hasSearched,
        noResults,
        lastQuery,
        removedMovies,
      })
    );
  };

  const handleSearch = async (query) => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    setLastQuery(trimmedQuery);
    setRemovedMovies([]);
    setErrorMessage("");
    setNoResults(false);
    setHasSearched(true);

    if (!API_BASE_URL) {
      setMovies([]);
      setErrorMessage(
        "API adresi tanımlı değil. VITE_API_URL değerini kontrol et."
      );
      return;
    }

    setLoading(true);
    setShowWarmupMessage(false);
    setMovies([]);

    const warmupTimer = setTimeout(() => {
      setShowWarmupMessage(true);
    }, WARMUP_MESSAGE_DELAY);

    try {
      const response = await axios.get(`${API_BASE_URL}/search`, {
        params: { q: trimmedQuery },
      });

      const results = Array.isArray(response?.data?.results)
        ? response.data.results
        : [];

      const formattedMovies = results.map((movie, index) => {
        const normalizedTurkishTitle =
          movie.turkish_title ??
          movie.title_tr ??
          movie.localized_title ??
          movie.translated_title ??
          "";

        const normalizedOriginalTitle =
          movie.original_title ??
          movie.originalTitle ??
          movie.original_name ??
          movie.name ??
          "";

        const normalizedOverviewTr =
          movie.overview_tr ??
          movie.overviewTr ??
          movie.overview ??
          "";

        const normalizedTaglineTr =
          movie.tagline_tr ??
          movie.taglineTr ??
          movie.tagline ??
          "";

        const normalizedWhyText = movie.why_text ?? movie.whyText ?? "";

        const normalizedYear =
          movie.release_year ??
          movie.year ??
          (movie.release_date ? String(movie.release_date).slice(0, 4) : "");

        return {
          ...movie,
          tmdb_id:
            movie.movie_id ?? movie.tmdb_id ?? movie.id ?? `movie-${index}`,
          poster_url: movie.poster_url
            ? movie.poster_url
            : movie.poster_path
              ? `${TMDB_IMAGE_BASE_URL}${movie.poster_path}`
              : "",
          year: normalizedYear,
          turkish_title: normalizedTurkishTitle,
          title: normalizedTurkishTitle || normalizedOriginalTitle,
          original_title: normalizedOriginalTitle,
          overview_tr: normalizedOverviewTr,
          tagline_tr: normalizedTaglineTr,
          why_text: normalizedWhyText,
        };
      });

      setMovies(formattedMovies);
      setNoResults(formattedMovies.length === 0);
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

  const handleRemoveMovie = (movieToRemove, indexToRemove) => {
    const updatedMovies = movies.filter(
      (movie) => movie.tmdb_id !== movieToRemove.tmdb_id
    );

    const removedItem = {
      ...movieToRemove,
      removedIndex: indexToRemove,
    };

    setMovies(updatedMovies);
    setRemovedMovies((prev) => [removedItem, ...prev]);
    setNoResults(updatedMovies.length === 0);
  };

  const handleRestoreLastRemoved = () => {
    if (removedMovies.length === 0) return;

    const [lastRemoved, ...remainingRemoved] = removedMovies;

    const targetIndex =
      typeof lastRemoved.removedIndex === "number"
        ? Math.min(lastRemoved.removedIndex, movies.length)
        : 0;

    const restoredMovie = { ...lastRemoved };
    delete restoredMovie.removedIndex;

    const restoredMovies = [...movies];
    restoredMovies.splice(targetIndex, 0, restoredMovie);

    setMovies(restoredMovies);
    setRemovedMovies(remainingRemoved);
    setNoResults(false);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const restoredRef = movieRefs.current[restoredMovie.tmdb_id];
        if (restoredRef) {
          restoredRef.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
        }
      });
    });
  };

  const resultCountText =
    movies.length > 0
      ? `${movies.length} film bulundu`
      : hasSearched && noResults
        ? "Sonuç bulunamadı"
        : "Aramaya hazır";

  const showIntro = !hasSearched && !loading && !lastQuery && movies.length === 0;

  return (
    <div style={styles.page}>
      <div style={styles.pageGlowTop} />
      <div style={styles.pageGlowRight} />
      <div style={styles.pageGrid} />

      <div style={styles.container}>
        {showIntro && (
          <section style={styles.heroSection}>
            <div style={styles.heroOverlay} />

            <div style={styles.heroContent}>
              <p style={styles.eyebrow}>Cue Smart Movie Recommendation</p>

              <div style={styles.exampleWrap}>
                <span style={styles.exampleLabel}>Hızlı başlangıç</span>

                <div style={styles.exampleList}>
                  {EXAMPLE_QUERIES.map((item) => (
                    <button
                      key={item}
                      type="button"
                      style={styles.exampleChip}
                      onClick={() => handleSearch(item)}
                      disabled={loading}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </section>
        )}

        <div
          ref={searchAnchorRef}
          style={{
            ...styles.searchAnchor,
            height: isSearchPinned ? `${searchBarHeight + 18}px` : "0px",
          }}
        />

        <div
          ref={searchShellRef}
          style={{
            ...styles.stickySearchWrap,
            ...(isSearchPinned ? styles.stickySearchWrapPinned : {}),
          }}
        >
          <div style={styles.stickySearchBackdrop} />
          <div style={styles.stickySearchInner}>
            <SearchBar
              onSearch={handleSearch}
              loading={loading}
              initialValue={lastQuery}
            />
          </div>
        </div>

        <section style={styles.resultsSection}>
          <div style={styles.resultsHeader}>
            <div style={styles.resultsHeaderLeft}>
              <h2 style={styles.resultsTitle}>Film Sonuçları</h2>

              {lastQuery ? (
                <p style={styles.resultsSubtitle}>
                  Son sorgu: <span style={styles.resultsQuery}>{lastQuery}</span>
                </p>
              ) : (
                <p style={styles.resultsSubtitle}>
                  Film, tür, ruh hali veya dönem yazarak aramaya başlayabilirsin.
                </p>
              )}
            </div>

            <div style={styles.resultsHeaderRight}>
              <span style={styles.countBadge}>{resultCountText}</span>
            </div>
          </div>

          {!loading && errorMessage ? (
            <div style={styles.inlineErrorBox}>{errorMessage}</div>
          ) : loading ? (
            <SearchLoadingState showWarmupMessage={showWarmupMessage} />
          ) : noResults ? (
            <div style={styles.emptyStateCard}>
              <div style={styles.emptyStateIcon}>🎬</div>
              <h3 style={styles.emptyStateTitle}>Sonuç bulunamadı</h3>
              <p style={styles.emptyStateText}>
                Daha genel bir ifade dene ya da tür, dönem ve ruh hali gibi
                ipuçları ekle.
              </p>
            </div>
          ) : movies.length > 0 ? (
            <div style={styles.grid}>
              {movies.map((movie, index) => (
                <div
                  key={movie.tmdb_id}
                  ref={(el) => {
                    if (el) {
                      movieRefs.current[movie.tmdb_id] = el;
                    } else {
                      delete movieRefs.current[movie.tmdb_id];
                    }
                  }}
                >
                  <MovieCard
                    movie={movie}
                    onRemove={() => handleRemoveMovie(movie, index)}
                    onOpenDetail={persistStateForDetail}
                  />
                </div>
              ))}
            </div>
          ) : (
            <div style={styles.emptyStateCard}>
              <div style={styles.emptyStateIcon}>🔎</div>
              <h3 style={styles.emptyStateTitle}>Henüz arama yapılmadı</h3>
              <p style={styles.emptyStateText}>
                Örnek: “90'larda geçen karanlık bir gerilim filmi”
              </p>
            </div>
          )}
        </section>
      </div>

      {removedMovies.length > 0 && !loading && (
        <div style={styles.floatingUndo}>
          <div style={styles.floatingUndoTextWrap}>
            <span style={styles.floatingUndoLabel}>Film kaldırıldı</span>
            <strong style={styles.floatingUndoTitle}>
              {removedMovies[0]?.turkish_title ||
                removedMovies[0]?.title ||
                removedMovies[0]?.original_title ||
                "Son kaldırılan film"}
            </strong>
          </div>

          <button
            type="button"
            style={styles.floatingUndoButton}
            onClick={handleRestoreLastRemoved}
          >
            Geri Al
          </button>
        </div>
      )}
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
    position: "relative",
    minHeight: "100vh",
    width: "100%",
    overflow: "visible",
    boxSizing: "border-box",
    background: `
      radial-gradient(circle at 12% 10%, rgba(99,102,241,0.18), transparent 22%),
      radial-gradient(circle at 88% 16%, rgba(168,85,247,0.18), transparent 24%),
      radial-gradient(circle at 80% 72%, rgba(59,130,246,0.10), transparent 20%),
      linear-gradient(180deg, #030712 0%, #050816 42%, #040814 100%)
    `,
    color: "#f8fafc",
    padding: "22px 20px 46px",
  },

  pageGlowTop: {
    position: "absolute",
    top: "-120px",
    left: "-120px",
    width: "420px",
    height: "420px",
    borderRadius: "50%",
    background: "rgba(99,102,241,0.14)",
    filter: "blur(90px)",
    pointerEvents: "none",
  },

  pageGlowRight: {
    position: "absolute",
    top: "80px",
    right: "-120px",
    width: "420px",
    height: "420px",
    borderRadius: "50%",
    background: "rgba(168,85,247,0.16)",
    filter: "blur(100px)",
    pointerEvents: "none",
  },

  pageGrid: {
    position: "absolute",
    inset: 0,
    backgroundImage:
      "linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)",
    backgroundSize: "64px 64px",
    maskImage: "linear-gradient(180deg, rgba(0,0,0,0.55), rgba(0,0,0,0))",
    pointerEvents: "none",
  },

  container: {
    position: "relative",
    zIndex: 1,
    width: "100%",
    maxWidth: "1240px",
    margin: "0 auto",
    boxSizing: "border-box",
  },

  searchAnchor: {
    width: "100%",
  },

  heroSection: {
    position: "relative",
    width: "100%",
    overflow: "hidden",
    boxSizing: "border-box",
    borderRadius: "30px",
    marginBottom: "18px",
    border: "1px solid rgba(148,163,184,0.10)",
    background:
      "linear-gradient(135deg, rgba(5,10,24,0.96) 0%, rgba(8,11,28,0.92) 45%, rgba(17,24,39,0.86) 100%)",
    boxShadow: "0 24px 60px rgba(0,0,0,0.34)",
  },

  heroOverlay: {
    position: "absolute",
    inset: 0,
    background: `
      radial-gradient(circle at top right, rgba(99,102,241,0.16), transparent 28%),
      radial-gradient(circle at left center, rgba(168,85,247,0.08), transparent 34%)
    `,
    pointerEvents: "none",
  },

  heroContent: {
    position: "relative",
    zIndex: 1,
    padding: "28px 30px 24px",
  },

  eyebrow: {
    margin: "0 0 18px",
    color: "#e2e8f0",
    fontSize: "clamp(1.05rem, 1.8vw, 1.5rem)",
    fontWeight: "900",
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    lineHeight: "1.3",
    textShadow: "0 4px 18px rgba(59,130,246,0.18)",
  },

  exampleWrap: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },

  exampleLabel: {
    color: "#cbd5e1",
    fontSize: "14px",
    fontWeight: "700",
  },

  exampleList: {
    display: "flex",
    flexWrap: "wrap",
    gap: "10px",
    maxWidth: "820px",
  },

  exampleChip: {
    padding: "11px 15px",
    borderRadius: "999px",
    border: "1px solid rgba(129,140,248,0.14)",
    background: "rgba(10, 18, 36, 0.82)",
    color: "#e2e8f0",
    cursor: "pointer",
    fontSize: "13px",
    fontWeight: "700",
    lineHeight: "1.4",
    transition: "transform 0.15s ease, border-color 0.15s ease",
    boxShadow: "0 8px 18px rgba(0,0,0,0.14)",
  },

  stickySearchWrap: {
    position: "relative",
    width: "100%",
    marginBottom: "18px",
  },

  stickySearchWrapPinned: {
    position: "fixed",
    top: `${PIN_TOP_OFFSET}px`,
    left: "50%",
    transform: "translateX(-50%)",
    width: "min(1240px, calc(100vw - 40px))",
    zIndex: 220,
  },

  stickySearchBackdrop: {
    position: "absolute",
    inset: 0,
    borderRadius: "24px",
    background: "rgba(3, 7, 18, 0.72)",
    border: "1px solid rgba(148,163,184,0.08)",
    backdropFilter: "blur(16px)",
    WebkitBackdropFilter: "blur(16px)",
    boxShadow: "0 14px 34px rgba(0,0,0,0.24)",
  },

  stickySearchInner: {
    position: "relative",
    zIndex: 1,
    padding: "10px",
    borderRadius: "24px",
  },

  resultsSection: {
    width: "100%",
    boxSizing: "border-box",
    padding: "24px 26px 26px",
    borderRadius: "26px",
    background:
      "linear-gradient(180deg, rgba(8,13,28,0.78) 0%, rgba(7,12,26,0.62) 100%)",
    border: "1px solid rgba(148,163,184,0.08)",
    boxShadow: "0 12px 28px rgba(0,0,0,0.15)",
  },

  resultsHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: "18px",
    flexWrap: "wrap",
    marginBottom: "18px",
    paddingBottom: "16px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
  },

  resultsHeaderLeft: {
    minWidth: 0,
  },

  resultsHeaderRight: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    flexWrap: "wrap",
    justifyContent: "flex-end",
  },

  resultsTitle: {
    margin: 0,
    fontSize: "clamp(1.6rem, 2.2vw, 2rem)",
    fontWeight: "800",
    lineHeight: 1.12,
    letterSpacing: "-0.02em",
    color: "#ffffff",
  },

  resultsSubtitle: {
    margin: "10px 0 0",
    fontSize: "15px",
    color: "#cbd5e1",
    lineHeight: "1.55",
  },

  resultsQuery: {
    color: "#ffffff",
    fontWeight: "700",
  },

  countBadge: {
    display: "inline-flex",
    alignItems: "center",
    padding: "10px 14px",
    borderRadius: "999px",
    background: "rgba(59,130,246,0.10)",
    border: "1px solid rgba(96,165,250,0.14)",
    color: "#dbeafe",
    fontSize: "13px",
    fontWeight: "700",
  },

  loadingStateWrap: {
    display: "flex",
    flexDirection: "column",
    gap: "18px",
  },

  loadingStateTop: {
    display: "flex",
    alignItems: "center",
    gap: "16px",
    padding: "20px 18px",
    borderRadius: "18px",
    background:
      "linear-gradient(135deg, rgba(12,20,40,0.92) 0%, rgba(10,18,36,0.78) 100%)",
    border: "1px solid rgba(96,165,250,0.14)",
    boxShadow: "0 10px 28px rgba(0,0,0,0.18)",
  },

  loadingSpinner: {
    width: "46px",
    height: "46px",
    borderRadius: "50%",
    border: "4px solid rgba(148,163,184,0.18)",
    borderTop: "4px solid #60a5fa",
    flexShrink: 0,
    animation: "cueSpin 0.9s linear infinite",
  },

  loadingTextWrap: {
    minWidth: 0,
  },

  loadingTitle: {
    margin: "0 0 6px",
    color: "#f8fafc",
    fontSize: "18px",
    fontWeight: "800",
    lineHeight: "1.35",
    letterSpacing: "-0.01em",
  },

  loadingSubtitle: {
    margin: 0,
    color: "#cbd5e1",
    fontSize: "14px",
    lineHeight: "1.6",
    fontWeight: "500",
  },

  inlineInfoBox: {
    marginBottom: "0",
    padding: "14px 16px",
    borderRadius: "14px",
    background: "rgba(120,53,15,0.18)",
    border: "1px solid rgba(251,191,36,0.18)",
    color: "#fde68a",
    fontSize: "14px",
    lineHeight: "1.6",
    fontWeight: "600",
  },

  inlineErrorBox: {
    padding: "14px 16px",
    borderRadius: "14px",
    background: "rgba(127,29,29,0.18)",
    border: "1px solid rgba(248,113,113,0.20)",
    color: "#fecaca",
    fontSize: "14px",
    lineHeight: "1.6",
    fontWeight: "600",
  },

  emptyStateCard: {
    padding: "40px 22px",
    borderRadius: "18px",
    background: "rgba(7, 14, 28, 0.60)",
    border: "1px dashed rgba(148,163,184,0.16)",
    textAlign: "center",
  },

  emptyStateIcon: {
    fontSize: "30px",
    marginBottom: "10px",
  },

  emptyStateTitle: {
    margin: "0 0 8px",
    color: "#f8fafc",
    fontSize: "22px",
    fontWeight: "700",
    lineHeight: "1.3",
  },

  emptyStateText: {
    margin: 0,
    color: "#94a3b8",
    fontSize: "15px",
    lineHeight: "1.7",
  },

  grid: {
    display: "flex",
    flexDirection: "column",
    gap: "14px",
    width: "100%",
  },

  floatingUndo: {
    position: "fixed",
    right: "18px",
    bottom: "18px",
    zIndex: 260,
    display: "flex",
    alignItems: "center",
    gap: "12px",
    minWidth: "280px",
    maxWidth: "420px",
    padding: "14px 16px",
    borderRadius: "16px",
    background: "rgba(8, 16, 32, 0.94)",
    border: "1px solid rgba(96,165,250,0.18)",
    boxShadow: "0 18px 38px rgba(0,0,0,0.32)",
    backdropFilter: "blur(12px)",
    WebkitBackdropFilter: "blur(12px)",
  },

  floatingUndoTextWrap: {
    display: "flex",
    flexDirection: "column",
    minWidth: 0,
    flex: 1,
  },

  floatingUndoLabel: {
    color: "#93c5fd",
    fontSize: "12px",
    fontWeight: "700",
    marginBottom: "4px",
  },

  floatingUndoTitle: {
    color: "#f8fafc",
    fontSize: "14px",
    lineHeight: "1.4",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },

  floatingUndoButton: {
    flexShrink: 0,
    padding: "10px 14px",
    borderRadius: "10px",
    border: "1px solid rgba(96,165,250,0.22)",
    background: "rgba(26, 40, 68, 0.90)",
    color: "#bfdbfe",
    fontSize: "13px",
    fontWeight: "800",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },

  skeletonCard: {
    display: "grid",
    gridTemplateColumns: "118px 1fr",
    width: "100%",
    minHeight: "156px",
    borderRadius: "18px",
    overflow: "hidden",
    background:
      "linear-gradient(90deg, rgba(17,24,39,0.95) 0%, rgba(8,15,32,0.95) 100%)",
    border: "1px solid rgba(255,255,255,0.06)",
    boxShadow: "0 10px 20px rgba(0,0,0,0.14)",
  },

  skeletonPoster: {
    minHeight: "156px",
    ...shimmer,
  },

  skeletonBody: {
    padding: "16px 20px",
  },

  skeletonTitle: {
    height: "22px",
    width: "34%",
    borderRadius: "10px",
    marginBottom: "12px",
    ...shimmer,
  },

  skeletonMeta: {
    height: "15px",
    width: "18%",
    borderRadius: "10px",
    marginBottom: "14px",
    ...shimmer,
  },

  skeletonTextLong: {
    height: "15px",
    width: "84%",
    borderRadius: "10px",
    marginBottom: "9px",
    ...shimmer,
  },

  skeletonTextMedium: {
    height: "15px",
    width: "60%",
    borderRadius: "10px",
    ...shimmer,
  },
};

export default SearchPage;
