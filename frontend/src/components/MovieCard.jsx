import { useState } from "react";
import EmojiCurve from "./EmojiCurve";
import FeedbackButtons from "./FeedbackButtons";

const GENRE_MAP = {
  28: "Aksiyon", 12: "Macera", 16: "Animasyon", 35: "Komedi",
  80: "Suç", 99: "Belgesel", 18: "Drama", 10751: "Aile",
  14: "Fantastik", 36: "Tarih", 27: "Korku", 10402: "Müzik",
  9648: "Gizem", 10749: "Romantik", 878: "Bilim Kurgu",
  53: "Gerilim", 10752: "Savaş", 37: "Kovboy",
};

const GENRE_COLORS = {
  28: { bg: "rgba(220,38,38,0.15)", color: "#fca5a5", border: "rgba(220,38,38,0.25)" },
  27: { bg: "rgba(124,58,237,0.15)", color: "#c4b5fd", border: "rgba(124,58,237,0.25)" },
  53: { bg: "rgba(245,158,11,0.15)", color: "#fcd34d", border: "rgba(245,158,11,0.25)" },
  878: { bg: "rgba(6,182,212,0.15)", color: "#67e8f9", border: "rgba(6,182,212,0.25)" },
  18: { bg: "rgba(59,130,246,0.15)", color: "#93c5fd", border: "rgba(59,130,246,0.25)" },
  35: { bg: "rgba(234,179,8,0.15)", color: "#fde047", border: "rgba(234,179,8,0.25)" },
  10749: { bg: "rgba(236,72,153,0.15)", color: "#f9a8d4", border: "rgba(236,72,153,0.25)" },
  default: { bg: "rgba(148,163,184,0.12)", color: "#cbd5e1", border: "rgba(148,163,184,0.2)" },
};

function GenreBadge({ id }) {
  const name = GENRE_MAP[id];
  if (!name) return null;
  const c = GENRE_COLORS[id] || GENRE_COLORS.default;
  return (
    <span style={{
      padding: "3px 10px",
      borderRadius: "999px",
      fontSize: "11px",
      fontWeight: "600",
      background: c.bg,
      color: c.color,
      border: `1px solid ${c.border}`,
    }}>
      {name}
    </span>
  );
}

function MovieCard({ movie }) {
  const [hovered, setHovered] = useState(false);
  const palette = Array.isArray(movie.color_palette) ? movie.color_palette : [];
  const genres = Array.isArray(movie.genre_ids) ? movie.genre_ids.slice(0, 3) : [];
  const hasPoster = Boolean(movie.poster_url);

  return (
    <article
      style={{
        ...styles.card,
        ...(hovered ? styles.cardHovered : {}),
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div style={styles.posterWrapper}>
        {hasPoster ? (
          <img
            src={movie.poster_url}
            alt={movie.title}
            style={styles.image}
            onError={(e) => { e.currentTarget.style.display = "none"; }}
          />
        ) : (
          <div style={styles.noPoster}>
            <span style={styles.noPosterText}>Poster Yok</span>
          </div>
        )}
        <div style={styles.posterOverlay} />
        {genres.length > 0 && (
          <div style={styles.genreOverlay}>
            {genres.map((id) => <GenreBadge key={id} id={id} />)}
          </div>
        )}
      </div>

      <div style={styles.content}>
        <div style={styles.headerRow}>
          <h3 style={styles.title}>
            {movie.title} {movie.year ? `(${movie.year})` : ""}
          </h3>
          <span style={styles.scoreBadge}>⭐ {movie.vote_average}</span>
        </div>

        <p style={styles.overview}>{movie.overview}</p>

        <div style={styles.reasonBox}>
          <p style={styles.reasonLabel}>Neden önerildi?</p>
          <p style={styles.why}>{movie.why_text}</p>
        </div>

        <EmojiCurve curve={movie.emotion_curve} />

        {palette.length > 0 && (
          <div style={styles.paletteWrapper}>
            <p style={styles.paletteLabel}>Renk Paleti</p>
            <div style={styles.paletteRow}>
              {palette.map((color, index) => (
                <div
                  key={`${color}-${index}`}
                  style={{ ...styles.colorSwatch, backgroundColor: color }}
                  title={color}
                />
              ))}
            </div>
          </div>
        )}

        <FeedbackButtons filmId={movie.tmdb_id} />
      </div>
    </article>
  );
}

const styles = {
  card: {
    background: "linear-gradient(180deg, rgba(31,41,55,1) 0%, rgba(17,24,39,1) 100%)",
    borderRadius: "20px",
    overflow: "hidden",
    boxShadow: "0 12px 30px rgba(0,0,0,0.28)",
    border: "1px solid rgba(255,255,255,0.06)",
    transition: "transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease",
    cursor: "pointer",
  },
  cardHovered: {
    transform: "translateY(-5px) scale(1.01)",
    border: "1px solid rgba(96,165,250,0.4)",
    boxShadow: "0 24px 48px rgba(0,0,0,0.45)",
  },
  posterWrapper: {
    position: "relative",
    overflow: "hidden",
  },
  image: {
    width: "100%",
    height: "260px",
    objectFit: "cover",
    display: "block",
    transition: "transform 0.3s ease",
  },
  posterOverlay: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    height: "80px",
    background: "linear-gradient(to top, rgba(17,24,39,0.95), transparent)",
    pointerEvents: "none",
  },
  genreOverlay: {
    position: "absolute",
    bottom: "10px",
    left: "12px",
    display: "flex",
    gap: "5px",
    flexWrap: "wrap",
  },
  noPoster: {
    width: "100%",
    height: "180px",
    background: "#111827",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
  },
  noPosterText: {
    color: "#9ca3af",
    fontSize: "16px",
    fontWeight: "600",
  },
  content: {
    padding: "18px",
  },
  headerRow: {
    display: "flex",
    justifyContent: "space-between",
    gap: "12px",
    alignItems: "flex-start",
    marginBottom: "12px",
  },
  title: {
    margin: 0,
    fontSize: "20px",
    lineHeight: "1.3",
    flex: 1,
  },
  scoreBadge: {
    flexShrink: 0,
    padding: "8px 10px",
    borderRadius: "999px",
    background: "rgba(251,191,36,0.14)",
    color: "#fcd34d",
    fontSize: "13px",
    fontWeight: "700",
    border: "1px solid rgba(251,191,36,0.18)",
  },
  overview: {
    color: "#cbd5e1",
    fontSize: "14px",
    lineHeight: "1.65",
    marginBottom: "16px",
  },
  reasonBox: {
    background: "rgba(59,130,246,0.08)",
    border: "1px solid rgba(96,165,250,0.14)",
    borderRadius: "14px",
    padding: "12px",
    marginBottom: "14px",
  },
  reasonLabel: {
    margin: "0 0 6px 0",
    fontSize: "12px",
    color: "#93c5fd",
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  },
  why: {
    margin: 0,
    color: "#dbeafe",
    fontSize: "14px",
    lineHeight: "1.55",
  },
  paletteWrapper: {
    marginTop: "16px",
  },
  paletteLabel: {
    fontSize: "13px",
    color: "#d1d5db",
    marginBottom: "8px",
  },
  paletteRow: {
    display: "flex",
    gap: "8px",
    flexWrap: "wrap",
  },
  colorSwatch: {
    width: "28px",
    height: "28px",
    borderRadius: "999px",
    border: "1px solid rgba(255,255,255,0.2)",
  },
};

export default MovieCard;
