function MovieCard({ movie }) {
  const palette = Array.isArray(movie.color_palette) ? movie.color_palette : [];
  const hasPoster = Boolean(movie.poster_url);

  return (
    <div style={styles.card}>
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
        <h3 style={styles.title}>
          {movie.title} {movie.year ? `(${movie.year})` : ""}
        </h3>

        <p style={styles.rating}>⭐ {movie.vote_average}</p>
        <p style={styles.text}>{movie.overview}</p>
        <p style={styles.why}>{movie.why_text}</p>

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
      </div>
    </div>
  );
}

const styles = {
  card: {
    background: "#1f2937",
    borderRadius: "16px",
    overflow: "hidden",
    boxShadow: "0 10px 25px rgba(0,0,0,0.25)"
  },
  image: {
    width: "100%",
    height: "320px",
    objectFit: "cover",
    display: "block"
  },
  noPoster: {
    width: "100%",
    height: "220px",
    background: "#111827",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderBottom: "1px solid rgba(255,255,255,0.08)"
  },
  noPosterText: {
    color: "#9ca3af",
    fontSize: "18px",
    fontWeight: "600"
  },
  content: {
    padding: "16px"
  },
  title: {
    marginBottom: "8px"
  },
  rating: {
    marginBottom: "10px",
    color: "#fbbf24"
  },
  text: {
    color: "#d1d5db",
    fontSize: "14px",
    lineHeight: "1.5"
  },
  why: {
    marginTop: "12px",
    color: "#93c5fd",
    fontSize: "14px"
  },
  paletteWrapper: {
    marginTop: "16px"
  },
  paletteLabel: {
    fontSize: "13px",
    color: "#d1d5db",
    marginBottom: "8px"
  },
  paletteRow: {
    display: "flex",
    gap: "8px"
  },
  colorSwatch: {
    width: "28px",
    height: "28px",
    borderRadius: "999px",
    border: "1px solid rgba(255,255,255,0.2)"
  }
};

export default MovieCard;
