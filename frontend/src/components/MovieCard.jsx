import { useState } from "react";
import { useNavigate } from "react-router-dom";

function ImagePlaceholderIcon() {
  return (
    <svg
      width="38"
      height="38"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <rect
        x="3"
        y="4"
        width="18"
        height="16"
        rx="2"
        stroke="#94a3b8"
        strokeWidth="1.8"
      />
      <circle cx="9" cy="10" r="1.7" fill="#94a3b8" />
      <path
        d="M5.5 17l4.2-4.2a1 1 0 011.4 0l2.1 2.1a1 1 0 001.4 0l1.2-1.2a1 1 0 011.4 0L18.5 15"
        stroke="#94a3b8"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function MovieCard({ movie, onRemove, onOpenDetail }) {
  const [hovered, setHovered] = useState(false);
  const [titleHovered, setTitleHovered] = useState(false);
  const navigate = useNavigate();

  const displayTitle =
    movie?.turkish_title ||
    movie?.title ||
    movie?.original_title ||
    movie?.english_title ||
    movie?.name ||
    "İsimsiz Film";

  const displaySubtitle =
    movie?.original_title && movie?.original_title !== displayTitle
      ? movie.original_title
      : "";

  const displayOverview =
    movie?.tagline_tr?.trim() ||
    movie?.overview_tr?.trim() ||
    movie?.why_text?.trim() ||
    movie?.tagline?.trim() ||
    movie?.overview?.trim() ||
    movie?.description?.trim() ||
    movie?.summary?.trim() ||
    "Bu film için açıklama bilgisi bulunamadı.";

  const detailId = movie?.tmdb_id || movie?.movie_id || movie?.id;

  const formattedDate = movie?.release_date
    ? new Date(movie.release_date).toLocaleDateString("tr-TR", {
        day: "numeric",
        month: "long",
        year: "numeric",
      })
    : movie?.year || "Tarih bilgisi yok";

  const handleOpenDetail = () => {
    onOpenDetail?.();

    if (!detailId) return;

    navigate(`/movie/${detailId}`, {
      state: { movie },
    });
  };

  return (
    <article
      style={{
        ...styles.card,
        ...(hovered ? styles.cardHovered : {}),
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div style={styles.posterColumn}>
        {movie?.poster_url ? (
          <>
            <img
              src={movie.poster_url}
              alt={displayTitle}
              style={styles.image}
              onError={(e) => {
                e.currentTarget.style.display = "none";
                const fallback = e.currentTarget.nextSibling;
                if (fallback) fallback.style.display = "flex";
              }}
            />
            <div style={{ ...styles.noPoster, display: "none" }}>
              <ImagePlaceholderIcon />
            </div>
          </>
        ) : (
          <div style={styles.noPoster}>
            <ImagePlaceholderIcon />
          </div>
        )}
      </div>

      <div style={styles.contentColumn}>
        <div style={styles.topSection}>
          <div style={styles.headerBlock}>
            <button
              type="button"
              style={{
                ...styles.titleButton,
                ...(titleHovered ? styles.titleButtonHovered : {}),
              }}
              onClick={handleOpenDetail}
              onMouseEnter={() => setTitleHovered(true)}
              onMouseLeave={() => setTitleHovered(false)}
            >
              {displayTitle}
            </button>

            {displaySubtitle ? (
              <p style={styles.subtitle}>{displaySubtitle}</p>
            ) : null}

            <div style={styles.metaRow}>
              <span style={styles.dateBadge}>{formattedDate}</span>
            </div>
          </div>

          <p style={styles.overview}>{displayOverview}</p>
        </div>

        <div style={styles.actionsRow}>
          <button
            type="button"
            style={styles.detailButton}
            onClick={handleOpenDetail}
            disabled={!detailId}
          >
            Detaya Git
          </button>

          <button
            type="button"
            style={styles.removeButton}
            onClick={onRemove}
          >
            Kaldır
          </button>
        </div>
      </div>
    </article>
  );
}

const styles = {
  card: {
    display: "grid",
    gridTemplateColumns: "118px 1fr",
    alignItems: "stretch",
    background:
      "linear-gradient(180deg, rgba(10,17,33,0.98) 0%, rgba(7,13,27,0.96) 100%)",
    borderRadius: "18px",
    overflow: "hidden",
    border: "1px solid rgba(148,163,184,0.08)",
    boxShadow: "0 10px 22px rgba(0,0,0,0.14)",
    transition:
      "transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease",
  },

  cardHovered: {
    transform: "translateY(-2px)",
    border: "1px solid rgba(96,165,250,0.18)",
    boxShadow: "0 14px 28px rgba(0,0,0,0.18)",
  },

  posterColumn: {
    background: "#dbe4ee",
    minHeight: "156px",
    position: "relative",
  },

  image: {
    width: "100%",
    height: "100%",
    minHeight: "156px",
    objectFit: "cover",
    display: "block",
  },

  noPoster: {
    width: "100%",
    minHeight: "156px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#dbe4ee",
  },

  contentColumn: {
    padding: "16px 20px 14px 20px",
    minWidth: 0,
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between",
    gap: "14px",
    boxSizing: "border-box",
  },

  topSection: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    minWidth: 0,
  },

  headerBlock: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
    minWidth: 0,
  },

  titleButton: {
    border: "none",
    background: "transparent",
    padding: 0,
    margin: 0,
    color: "#f8fafc",
    fontSize: "clamp(1.08rem, 1.65vw, 1.4rem)",
    fontWeight: "800",
    lineHeight: "1.15",
    textAlign: "left",
    cursor: "pointer",
    letterSpacing: "-0.03em",
  },

  titleButtonHovered: {
    color: "#93c5fd",
  },

  subtitle: {
    margin: 0,
    color: "#7f90aa",
    fontSize: "12.5px",
    lineHeight: "1.35",
    fontWeight: "500",
    letterSpacing: "0.01em",
  },

  metaRow: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    flexWrap: "wrap",
    marginTop: "4px",
  },

  dateBadge: {
    display: "inline-flex",
    alignItems: "center",
    padding: "6px 10px",
    borderRadius: "999px",
    background: "rgba(59,130,246,0.10)",
    border: "1px solid rgba(96,165,250,0.14)",
    color: "#dbeafe",
    fontSize: "12px",
    fontWeight: "700",
    lineHeight: "1.2",
  },

  overview: {
    margin: 0,
    color: "#d4deea",
    fontSize: "14px",
    lineHeight: "1.66",
    maxWidth: "88%",
    display: "-webkit-box",
    WebkitLineClamp: 3,
    WebkitBoxOrient: "vertical",
    overflow: "hidden",
  },

  actionsRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    flexWrap: "wrap",
  },

  detailButton: {
    padding: "8px 12px",
    borderRadius: "10px",
    border: "1px solid rgba(96,165,250,0.16)",
    background: "rgba(26, 40, 68, 0.84)",
    color: "#bfdbfe",
    fontSize: "12.5px",
    fontWeight: "700",
    cursor: "pointer",
  },

  removeButton: {
    padding: "8px 12px",
    borderRadius: "10px",
    border: "1px solid rgba(248,113,113,0.16)",
    background: "rgba(127,29,29,0.12)",
    color: "#fecaca",
    fontSize: "12.5px",
    fontWeight: "700",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
};

export default MovieCard;
