import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import axios from "axios";
import FeedbackModal from "../components/FeedbackModal";

const API_BASE_URL = import.meta.env.VITE_API_URL;
const TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500";

function ExternalLinkIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M14 5h5v5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M10 14L19 5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M19 14v4a1 1 0 01-1 1H6a1 1 0 01-1-1V6a1 1 0 011-1h4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M8 6.5v11l9-5.5-9-5.5z" />
    </svg>
  );
}

function BackIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M15 18l-6-6 6-6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ClockIcon() {
  return (
    <span aria-hidden="true" style={styles.metaEmoji}>
      ⏱
    </span>
  );
}

function StarIcon() {
  return (
    <span aria-hidden="true" style={styles.metaEmoji}>
      ⭐
    </span>
  );
}

function GlobeIcon() {
  return (
    <span aria-hidden="true" style={styles.metaEmoji}>
      🌍
    </span>
  );
}

function EmotionCurve({ points = [] }) {
  if (!Array.isArray(points) || points.length < 2) return null;

  const width = 100;
  const height = 36;

  const normalizedPoints = points.map((value, index) => {
    const numericValue = Number(value);
    const safeValue = Number.isFinite(numericValue) ? numericValue : 0;
    const clampedValue = Math.max(0, Math.min(1, safeValue));
    const x = (index / (points.length - 1)) * width;
    const y = height - clampedValue * height;

    return `${x},${y}`;
  });

  return (
    <div style={styles.curveSection}>
      <div style={styles.curveHeader}>
        <span style={styles.sectionLabel}>Duygu Eğrisi</span>
      </div>

      <div style={styles.curveCard}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          style={styles.curveSvg}
          aria-label="Duygu eğrisi grafiği"
        >
          <defs>
            <linearGradient
              id="emotionCurveStroke"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="0%"
            >
              <stop offset="0%" stopColor="#60a5fa" />
              <stop offset="50%" stopColor="#818cf8" />
              <stop offset="100%" stopColor="#22c55e" />
            </linearGradient>
          </defs>

          <line
            x1="0"
            y1={height}
            x2={width}
            y2={height}
            stroke="rgba(148,163,184,0.25)"
            strokeWidth="0.8"
          />

          <polyline
            fill="none"
            stroke="url(#emotionCurveStroke)"
            strokeWidth="2.5"
            strokeLinejoin="round"
            strokeLinecap="round"
            points={normalizedPoints.join(" ")}
          />
        </svg>

        <div style={styles.curveLegend}>
          <span style={styles.curveLegendText}>Başlangıç</span>
          <span style={styles.curveLegendText}>Final</span>
        </div>
      </div>
    </div>
  );
}

function MovieDetailPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { id } = useParams();

  const stateMovie = location.state?.movie || null;

  const [movie, setMovie] = useState(stateMovie);
  const [loading, setLoading] = useState(!stateMovie && !!id);
  const [error, setError] = useState("");
  const [isFeedbackOpen, setIsFeedbackOpen] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState("");

  useEffect(() => {
    const fetchMovieIfNeeded = async () => {
      if (stateMovie || !id || !API_BASE_URL) return;

      setLoading(true);
      setError("");

      try {
        const response = await axios.get(`${API_BASE_URL}/search`, {
          params: { q: id },
        });

        const results = Array.isArray(response?.data?.results)
          ? response.data.results
          : [];

        const matchedMovie =
          results.find(
            (item) =>
              String(item.movie_id ?? item.tmdb_id ?? item.id) === String(id)
          ) || null;

        if (matchedMovie) {
          setMovie({
            ...matchedMovie,
            tmdb_id:
              matchedMovie.movie_id ??
              matchedMovie.tmdb_id ??
              matchedMovie.id ??
              id,
            poster_url: matchedMovie.poster_url
              ? matchedMovie.poster_url
              : matchedMovie.poster_path
                ? `${TMDB_IMAGE_BASE_URL}${matchedMovie.poster_path}`
                : "",
            why_text: matchedMovie.why_text ?? matchedMovie.whyText ?? "",
            emotion_curve:
              matchedMovie.emotion_curve ??
              matchedMovie.emotionCurve ??
              [],
          });
        } else {
          setError("Film detayı bulunamadı.");
        }
      } catch (fetchError) {
        console.error("Movie detail fetch error:", fetchError);
        setError("Film detayı alınırken bir hata oluştu.");
      } finally {
        setLoading(false);
      }
    };

    fetchMovieIfNeeded();
  }, [API_BASE_URL, id, stateMovie]);

  const displayTitle = useMemo(() => {
    return (
      movie?.turkish_title ||
      movie?.title ||
      movie?.original_title ||
      movie?.english_title ||
      "İsimsiz Film"
    );
  }, [movie]);

  const displaySubtitle = useMemo(() => {
    if (!movie?.original_title) return "";
    return movie.original_title !== displayTitle ? movie.original_title : "";
  }, [movie, displayTitle]);

  const directorList = useMemo(() => {
    const raw =
      movie?.directors ??
      movie?.director ??
      movie?.director_name ??
      null;

    if (!raw) return [];

    if (Array.isArray(raw)) {
      return raw
        .map((item) => {
          if (typeof item === "string") return item.trim();
          if (item && typeof item === "object" && item.name) {
            return String(item.name).trim();
          }
          return "";
        })
        .filter(Boolean);
    }

    if (typeof raw === "string") {
      const trimmed = raw.trim();
      if (!trimmed) return [];

      try {
        const parsed = JSON.parse(trimmed);

        if (Array.isArray(parsed)) {
          return parsed
            .map((item) => {
              if (typeof item === "string") return item.trim();
              if (item && typeof item === "object" && item.name) {
                return String(item.name).trim();
              }
              return "";
            })
            .filter(Boolean);
        }

        if (parsed && typeof parsed === "object" && parsed.name) {
          return [String(parsed.name).trim()];
        }
      } catch {
        return [trimmed];
      }

      return [trimmed];
    }

    if (typeof raw === "object" && raw?.name) {
      return [String(raw.name).trim()];
    }

    return [];
  }, [movie]);

  const directorText = useMemo(() => {
    return directorList.join(", ");
  }, [directorList]);

  const directorLabelText = useMemo(() => {
    return directorList.length > 1 ? "Yönetmenler:" : "Yönetmen:";
  }, [directorList]);

  const displayWhyText = useMemo(() => {
    return (movie?.why_text ?? movie?.whyText ?? "").trim();
  }, [movie]);

  const displayOverview = useMemo(() => {
    return (
      movie?.overview_tr?.trim() ||
      movie?.overview?.trim() ||
      "Bu film için açıklama bilgisi bulunamadı."
    );
  }, [movie]);

  const emotionCurve = useMemo(() => {
    const raw =
      movie?.emotion_curve ??
      movie?.emotionCurve ??
      movie?.sentiment_curve ??
      movie?.sentimentCurve ??
      [];

    return Array.isArray(raw) ? raw : [];
  }, [movie]);

  const posterUrl = useMemo(() => {
    if (movie?.poster_url) return movie.poster_url;
    if (movie?.poster_path) return `${TMDB_IMAGE_BASE_URL}${movie.poster_path}`;
    return "";
  }, [movie]);

  const releaseYear = useMemo(() => {
    if (movie?.release_year) return movie.release_year;
    if (movie?.release_date) return String(movie.release_date).slice(0, 4);
    return "";
  }, [movie]);

  const runtimeText = useMemo(() => {
    return movie?.runtime_formatted || movie?.runtime || "";
  }, [movie]);

  const scoreText = useMemo(() => {
    return movie?.imdb_score ? String(movie.imdb_score) : "";
  }, [movie]);

  const languageCode = useMemo(() => {
    return movie?.original_language?.toUpperCase?.() || "";
  }, [movie]);

  const genres = useMemo(() => {
    if (Array.isArray(movie?.genres_tr) && movie.genres_tr.length > 0) {
      return movie.genres_tr;
    }
    return [];
  }, [movie]);

  const palette = useMemo(() => {
    const rawPalette = Array.isArray(movie?.color_palette)
      ? movie.color_palette
      : [];

    const fallback = {
      primary: "rgba(59,130,246,0.10)",
      secondary: "rgba(99,102,241,0.08)",
      tertiary: "rgba(148,163,184,0.08)",
      chip: "rgba(59,130,246,0.10)",
      chipBorder: "rgba(96,165,250,0.14)",
      whyBg: "rgba(59,130,246,0.08)",
      whyBorder: "rgba(96,165,250,0.12)",
      curveBg: "rgba(15,23,42,0.42)",
      curveBorder: "rgba(96,165,250,0.14)",
    };

    if (rawPalette.length === 0) return fallback;

    const [p1, p2, p3] = rawPalette;

    return {
      primary: p1 ? convertColorToAlpha(p1, 0.18) : fallback.primary,
      secondary: p2 ? convertColorToAlpha(p2, 0.14) : fallback.secondary,
      tertiary: p3 ? convertColorToAlpha(p3, 0.10) : fallback.tertiary,
      chip: p2 ? convertColorToAlpha(p2, 0.14) : fallback.chip,
      chipBorder: p2 ? convertColorToAlpha(p2, 0.22) : fallback.chipBorder,
      whyBg: p1 ? convertColorToAlpha(p1, 0.12) : fallback.whyBg,
      whyBorder: p2 ? convertColorToAlpha(p2, 0.18) : fallback.whyBorder,
      curveBg: p3 ? convertColorToAlpha(p3, 0.16) : fallback.curveBg,
      curveBorder: p2 ? convertColorToAlpha(p2, 0.20) : fallback.curveBorder,
    };
  }, [movie]);

  const dynamicPageBackground = useMemo(() => {
    return `
      radial-gradient(circle at top right, ${palette.primary}, transparent 24%),
      radial-gradient(circle at top left, ${palette.secondary}, transparent 22%),
      linear-gradient(180deg, #050915 0%, #071122 48%, #050915 100%)
    `;
  }, [palette]);

  const dynamicHeroBackground = useMemo(() => {
    return `
      radial-gradient(circle at 85% 18%, ${palette.secondary}, transparent 26%),
      radial-gradient(circle at 12% 82%, ${palette.tertiary}, transparent 24%),
      linear-gradient(180deg, rgba(8,13,28,0.82) 0%, rgba(7,12,26,0.66) 100%)
    `;
  }, [palette]);

  const handleBack = () => {
    navigate(-1);
  };

  const handleFeedbackSuccess = (action) => {
    const actionMap = {
      like: "Film beğenildi olarak kaydedildi.",
      dislike: "Film beğenilmedi olarak kaydedildi.",
      watched: "Film izlendi olarak kaydedildi.",
    };

    setFeedbackMessage(actionMap[action] || "Geri bildirim kaydedildi.");

    window.setTimeout(() => {
      setFeedbackMessage("");
    }, 2500);
  };

  if (loading) {
    return (
      <div style={{ ...styles.page, background: dynamicPageBackground }}>
        <div style={styles.container}>
          <button type="button" style={styles.backButton} onClick={handleBack}>
            <BackIcon />
            Geri Dön
          </button>

          <div style={styles.loadingCard}>Film detayı yükleniyor...</div>
        </div>
      </div>
    );
  }

  if (error || !movie) {
    return (
      <div style={{ ...styles.page, background: dynamicPageBackground }}>
        <div style={styles.container}>
          <button type="button" style={styles.backButton} onClick={handleBack}>
            <BackIcon />
            Geri Dön
          </button>

          <div style={styles.errorCard}>
            {error || "Film detayı bulunamadı."}
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <div style={{ ...styles.page, background: dynamicPageBackground }}>
        <div style={styles.container}>
          <button type="button" style={styles.backButton} onClick={handleBack}>
            <BackIcon />
            Sonuçlara Dön
          </button>

          <section
            style={{ ...styles.heroCard, background: dynamicHeroBackground }}
          >
            <div style={styles.posterArea}>
              {posterUrl ? (
                <img
                  src={posterUrl}
                  alt={displayTitle}
                  style={styles.posterImage}
                />
              ) : (
                <div style={styles.noPoster}>Poster yok</div>
              )}
            </div>

            <div style={styles.contentArea}>
              <div style={styles.headerBlock}>
                <div style={styles.titleRow}>
                  <h1 style={styles.title}>{displayTitle}</h1>
                  {releaseYear ? (
                    <span style={styles.titleYear}>({releaseYear})</span>
                  ) : null}
                </div>

                {displaySubtitle ? (
                  <p style={styles.subtitle}>{displaySubtitle}</p>
                ) : null}

                {directorList.length > 0 ? (
                  <p style={styles.directorLine}>
                    <span style={styles.directorLabel}>
                      {directorLabelText}
                    </span>
                    {directorText}
                  </p>
                ) : null}

                <div style={styles.inlineMetaRow}>
                  {runtimeText ? (
                    <span style={styles.inlineMetaItem}>
                      <ClockIcon />
                      {runtimeText}
                    </span>
                  ) : null}

                  {languageCode ? (
                    <span style={styles.inlineMetaItem}>
                      <GlobeIcon />
                      {languageCode}
                    </span>
                  ) : null}

                  {scoreText ? (
                    <span style={styles.scoreBadge}>
                      <StarIcon />
                      {scoreText}
                    </span>
                  ) : null}
                </div>
              </div>

              {genres.length > 0 ? (
                <div style={styles.genreSection}>
                  <span style={styles.sectionLabel}>Türler</span>
                  <div style={styles.genreList}>
                    {genres.map((genre) => (
                      <span
                        key={genre}
                        style={{
                          ...styles.genreBadge,
                          background: palette.chip,
                          border: `1px solid ${palette.chipBorder}`,
                        }}
                      >
                        {genre}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              {displayWhyText ? (
                <div style={styles.whySection}>
                  <span style={styles.sectionLabel}>Neden Önerildi?</span>
                  <p
                    style={{
                      ...styles.whyText,
                      background: palette.whyBg,
                      border: `1px solid ${palette.whyBorder}`,
                    }}
                  >
                    {displayWhyText}
                  </p>
                </div>
              ) : null}

              <div style={styles.descriptionSection}>
                <span style={styles.sectionLabel}>Özet</span>
                <p style={styles.overview}>{displayOverview}</p>
              </div>

              {emotionCurve.length > 1 ? (
                <div
                  style={{
                    ...styles.curveSectionWrap,
                    background: palette.curveBg,
                    border: `1px solid ${palette.curveBorder}`,
                  }}
                >
                  <EmotionCurve points={emotionCurve} />
                </div>
              ) : null}

              <div style={styles.actionsRow}>
                {movie?.youtube_url ? (
                  <a
                    href={movie.youtube_url}
                    target="_blank"
                    rel="noreferrer"
                    style={styles.primaryAction}
                  >
                    <PlayIcon />
                    Fragmanı Aç
                  </a>
                ) : null}

                {movie?.imdb_url ? (
                  <a
                    href={movie.imdb_url}
                    target="_blank"
                    rel="noreferrer"
                    style={styles.secondaryAction}
                  >
                    <ExternalLinkIcon />
                    IMDb Sayfası
                  </a>
                ) : null}

                <button
                  type="button"
                  style={styles.feedbackButton}
                  onClick={() => setIsFeedbackOpen(true)}
                >
                  Bu Film İçin Geri Bildirim Verebilirsin
                </button>
              </div>

              {feedbackMessage ? (
                <div style={styles.feedbackSuccessBox}>{feedbackMessage}</div>
              ) : null}
            </div>
          </section>
        </div>
      </div>

      <FeedbackModal
        open={isFeedbackOpen}
        onClose={() => setIsFeedbackOpen(false)}
        movieId={movie?.movie_id ?? movie?.tmdb_id ?? movie?.id}
        movieTitle={displayTitle}
        onSuccess={handleFeedbackSuccess}
      />
    </>
  );
}

function convertColorToAlpha(color, alpha) {
  if (!color) return `rgba(59,130,246,${alpha})`;

  if (color.startsWith("rgb(")) {
    const values = color
      .replace("rgb(", "")
      .replace(")", "")
      .split(",")
      .map((v) => v.trim());

    if (values.length === 3) {
      return `rgba(${values[0]}, ${values[1]}, ${values[2]}, ${alpha})`;
    }
  }

  if (color.startsWith("#")) {
    const hex = color.replace("#", "");

    const normalized =
      hex.length === 3
        ? hex
            .split("")
            .map((ch) => ch + ch)
            .join("")
        : hex;

    if (normalized.length === 6) {
      const r = parseInt(normalized.slice(0, 2), 16);
      const g = parseInt(normalized.slice(2, 4), 16);
      const b = parseInt(normalized.slice(4, 6), 16);

      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
  }

  return `rgba(59,130,246,${alpha})`;
}

const styles = {
  page: {
    minHeight: "100vh",
    width: "100%",
    boxSizing: "border-box",
    color: "#f8fafc",
    padding: "22px 20px 46px",
  },

  container: {
    width: "100%",
    maxWidth: "1240px",
    margin: "0 auto",
    boxSizing: "border-box",
  },

  backButton: {
    display: "inline-flex",
    alignItems: "center",
    gap: "8px",
    marginBottom: "16px",
    padding: "10px 14px",
    borderRadius: "999px",
    border: "1px solid rgba(148,163,184,0.12)",
    background: "rgba(8, 15, 30, 0.82)",
    color: "#dbeafe",
    fontSize: "14px",
    fontWeight: "700",
    cursor: "pointer",
  },

  heroCard: {
    display: "grid",
    gridTemplateColumns: "260px 1fr",
    gap: "24px",
    width: "100%",
    boxSizing: "border-box",
    padding: "24px",
    borderRadius: "26px",
    border: "1px solid rgba(148,163,184,0.08)",
    boxShadow: "0 12px 28px rgba(0,0,0,0.16)",
  },

  posterArea: {
    width: "100%",
  },

  posterImage: {
    width: "100%",
    borderRadius: "20px",
    objectFit: "cover",
    display: "block",
    boxShadow: "0 12px 28px rgba(0,0,0,0.22)",
  },

  noPoster: {
    width: "100%",
    minHeight: "390px",
    borderRadius: "20px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "rgba(148,163,184,0.12)",
    color: "#cbd5e1",
    fontWeight: "700",
  },

  contentArea: {
    minWidth: 0,
    display: "flex",
    flexDirection: "column",
    gap: "22px",
  },

  headerBlock: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },

  titleRow: {
    display: "flex",
    alignItems: "baseline",
    gap: "12px",
    flexWrap: "wrap",
  },

  title: {
    margin: 0,
    fontSize: "clamp(2rem, 3.2vw, 3rem)",
    fontWeight: "800",
    lineHeight: "1.03",
    letterSpacing: "-0.03em",
    color: "#ffffff",
  },

  titleYear: {
    color: "#94a3b8",
    fontSize: "clamp(1.1rem, 1.8vw, 1.5rem)",
    fontWeight: "700",
    lineHeight: "1.2",
  },

  subtitle: {
    margin: 0,
    color: "#a8b7cc",
    fontSize: "17px",
    lineHeight: "1.5",
    fontWeight: "600",
  },

  directorLine: {
    margin: 0,
    color: "#dbeafe",
    fontSize: "15px",
    lineHeight: "1.6",
    fontWeight: "600",
  },

  directorLabel: {
    color: "#93c5fd",
    fontWeight: "800",
    marginRight: "6px",
  },

  inlineMetaRow: {
    display: "flex",
    alignItems: "center",
    flexWrap: "wrap",
    gap: "12px 18px",
    marginTop: "2px",
  },

  inlineMetaItem: {
    display: "inline-flex",
    alignItems: "center",
    gap: "8px",
    color: "#dbeafe",
    fontSize: "14px",
    fontWeight: "700",
    lineHeight: "1.4",
  },

  metaEmoji: {
    fontSize: "15px",
    lineHeight: 1,
  },

  scoreBadge: {
    display: "inline-flex",
    alignItems: "center",
    gap: "8px",
    padding: "6px 10px",
    borderRadius: "999px",
    background: "rgba(250,204,21,0.12)",
    border: "1px solid rgba(250,204,21,0.20)",
    color: "#fde68a",
    fontSize: "14px",
    fontWeight: "800",
    lineHeight: "1.2",
  },

  genreSection: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },

  sectionLabel: {
    color: "#93c5fd",
    fontSize: "13px",
    fontWeight: "700",
    letterSpacing: "0.04em",
    textTransform: "uppercase",
  },

  genreList: {
    display: "flex",
    flexWrap: "wrap",
    gap: "10px",
  },

  genreBadge: {
    padding: "8px 12px",
    borderRadius: "999px",
    color: "#dbeafe",
    fontSize: "13px",
    fontWeight: "700",
  },

  whySection: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },

  whyText: {
    margin: 0,
    color: "#dbeafe",
    fontSize: "15px",
    lineHeight: "1.75",
    maxWidth: "88%",
    padding: "14px 16px",
    borderRadius: "16px",
  },

  descriptionSection: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },

  overview: {
    margin: 0,
    color: "#d4deea",
    fontSize: "16px",
    lineHeight: "1.8",
    maxWidth: "88%",
  },

  curveSectionWrap: {
    maxWidth: "88%",
    borderRadius: "18px",
    padding: "16px 18px",
  },

  curveSection: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },

  curveHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },

  curveCard: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },

  curveSvg: {
    width: "100%",
    height: "120px",
    display: "block",
  },

  curveLegend: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
  },

  curveLegendText: {
    color: "#a8b7cc",
    fontSize: "13px",
    fontWeight: "600",
  },

  actionsRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: "12px",
    marginTop: "2px",
  },

  primaryAction: {
    display: "inline-flex",
    alignItems: "center",
    gap: "8px",
    padding: "11px 16px",
    borderRadius: "12px",
    border: "1px solid rgba(99,102,241,0.24)",
    background: "linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%)",
    color: "#ffffff",
    textDecoration: "none",
    fontSize: "14px",
    fontWeight: "800",
    boxShadow: "0 10px 22px rgba(59,130,246,0.18)",
  },

  secondaryAction: {
    display: "inline-flex",
    alignItems: "center",
    gap: "8px",
    padding: "11px 16px",
    borderRadius: "12px",
    border: "1px solid rgba(148,163,184,0.14)",
    background: "rgba(8, 15, 30, 0.82)",
    color: "#dbeafe",
    textDecoration: "none",
    fontSize: "14px",
    fontWeight: "700",
  },

  feedbackButton: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "11px 16px",
    borderRadius: "12px",
    border: "1px solid rgba(34,197,94,0.18)",
    background: "rgba(34,197,94,0.12)",
    color: "#bbf7d0",
    fontSize: "14px",
    fontWeight: "800",
    cursor: "pointer",
  },

  feedbackSuccessBox: {
    padding: "12px 14px",
    borderRadius: "14px",
    background: "rgba(34,197,94,0.12)",
    border: "1px solid rgba(34,197,94,0.18)",
    color: "#bbf7d0",
    fontSize: "14px",
    fontWeight: "700",
    width: "fit-content",
  },

  loadingCard: {
    padding: "26px 22px",
    borderRadius: "20px",
    background: "rgba(8,13,28,0.78)",
    border: "1px solid rgba(148,163,184,0.08)",
    color: "#dbeafe",
    fontSize: "16px",
    fontWeight: "700",
  },

  errorCard: {
    padding: "26px 22px",
    borderRadius: "20px",
    background: "rgba(127,29,29,0.18)",
    border: "1px solid rgba(248,113,113,0.20)",
    color: "#fecaca",
    fontSize: "16px",
    fontWeight: "700",
  },
};

export default MovieDetailPage;
