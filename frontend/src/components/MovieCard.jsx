import { useState } from "react";
import { useNavigate } from "react-router-dom";

function ImagePlaceholderIcon() {
  return (
    <svg
      width="56"
      height="56"
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
        stroke="#9ca3af"
        strokeWidth="1.8"
      />
      <circle cx="9" cy="10" r="1.7" fill="#9ca3af" />
      <path
        d="M5.5 17l4.2-4.2a1 1 0 011.4 0l2.1 2.1a1 1 0 001.4 0l1.2-1.2a1 1 0 011.4 0L18.5 15"
        stroke="#9ca3af"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function MovieCard({ movie, onRemove }) {
  const [hovered, setHovered] = useState(false);
  const [titleHovered, setTitleHovered] = useState(false);
  const navigate = useNavigate();

  const handleOpenDetail = () => {
    navigate(`/movie/${movie.tmdb_id}`, {
      state: { movie },
    });
  };

  const formattedDate = movie.release_date
    ? new Date(movie.release_date).toLocaleDateString("tr-TR", {
        day: "numeric",
        month: "long",
        year: "numeric",
      })
    : movie.year || "Tarih bilgisi yok";

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
        {movie.poster_url ? (
          <>
            <img
              src={movie.poster_url}
              alt={movie.title}
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
        <div style={styles.topRow}>
          <div style={styles.titleArea}>
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
              {movie.title}
            </button>

            <p style={styles.dateText}>{formattedDate}</p>
          </div>

          <button
            type="button"
            style={styles.removeButton}
            onClick={() => onRemove(movie)}
          >
            Kaldır
          </button>
        </div>

        <p style={styles.overview}>
          {movie.overview || "Bu film için açıklama bilgisi bulunamadı."}
        </p>
      </div>
    </article>
  );
}

const styles = {
  card: {
    display: "grid",
    gridTemplateColumns: "180px 1fr",
    gap: "0",
    background:
      "linear-gradient(90deg, rgba(18,26,46,0.98) 0%, rgba(7,15,33,0.98) 100%)",
    borderRadius: "20px",
    overflow: "hidden",
    border: "1px solid rgba(255,255,255,0.08)",
    boxShadow: "0 10px 28px rgba(0,0,0,0.18)",
    transition:
      "transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease",
  },
  cardHovered: {
    transform: "translateY(-2px)",
    border: "1px solid rgba(148,163,184,0.24)",
    boxShadow: "0 16px 34px rgba(0,0,0,0.24)",
  },
  posterColumn: {
    background: "#e5e7eb",
    minHeight: "220px",
    position: "relative",
  },
  image: {
    width: "100%",
    height: "100%",
    minHeight: "220px",
    objectFit: "cover",
    display: "block",
  },
  noPoster: {
    width: "100%",
    minHeight: "220px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#e5e7eb",
  },
  contentColumn: {
    padding: "28px 30px",
    minWidth: 0,
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    boxSizing: "border-box",
  },
  topRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: "18px",
    marginBottom: "16px",
  },
  titleArea: {
    minWidth: 0,
    flex: 1,
    maxWidth: "980px",
  },
  titleButton: {
    border: "none",
    background: "transparent",
    padding: 0,
    margin: "0 0 10px 0",
    color: "#f8fafc",
    fontSize: "34px",
    fontWeight: "800",
    lineHeight: "1.15",
    textAlign: "left",
    cursor: "pointer",
    textDecoration: "none",
    letterSpacing: "-0.02em",
  },
  titleButtonHovered: {
    color: "#60a5fa",
    textDecoration: "underline",
    textUnderlineOffset: "4px",
  },
  dateText: {
    margin: 0,
    color: "#cbd5e1",
    fontSize: "17px",
    fontWeight: "500",
    lineHeight: "1.5",
  },
  removeButton: {
    flexShrink: 0,
    padding: "12px 18px",
    borderRadius: "12px",
    border: "1px solid rgba(248,113,113,0.28)",
    background: "rgba(127,29,29,0.22)",
    color: "#fecaca",
    fontSize: "15px",
    fontWeight: "800",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  overview: {
    margin: 0,
    color: "#e2e8f0",
    fontSize: "18px",
    lineHeight: "1.8",
    maxWidth: "980px",
    display: "-webkit-box",
    WebkitLineClamp: 4,
    WebkitBoxOrient: "vertical",
    overflow: "hidden",
  },
};

export default MovieCard;
