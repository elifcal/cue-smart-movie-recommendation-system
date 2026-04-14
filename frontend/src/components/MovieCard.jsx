import EmojiCurve from "./EmojiCurve";
import FeedbackButtons from "./FeedbackButtons";

function MovieCard({ movie }) {
  const palette = Array.isArray(movie.color_palette) ? movie.color_palette : [];
  const hasPoster = Boolean(movie.poster_url);

  return (
    <article style={styles.card}>
      {hasPoster ? (
        <img
          src={movie.poster_url}
          alt={movie.title}
          style={styles.image}
          onError={(e) => {
            e.currentTarget.style.display = "none";
          }}
        />
      ) : (
        <div style={styles.noPoster}>
          <span style={styles.noPosterText}>Poster Yok</span>
        </div>
      )}

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
  },
  image: {
    width: "100%",
    height: "260px",
    objectFit: "cover",
    display: "block",
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
    background: "rgba(251, 191, 36, 0.14)",
    color: "#fcd34d",
    fontSize: "13px",
    fontWeight: "700",
    border: "1px solid rgba(251, 191, 36, 0.18)",
  },
  overview: {
    color: "#cbd5e1",
    fontSize: "14px",
    lineHeight: "1.65",
    marginBottom: "16px",
  },
  reasonBox: {
    background: "rgba(59, 130, 246, 0.08)",
    border: "1px solid rgba(96, 165, 250, 0.14)",
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
