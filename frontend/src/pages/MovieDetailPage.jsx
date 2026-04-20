import { useLocation, useNavigate, useParams } from "react-router-dom";
import EmojiCurve from "../components/EmojiCurve";
import FeedbackButtons from "../components/FeedbackButtons";

const GENRE_MAP = {
  28: "Aksiyon",
  12: "Macera",
  16: "Animasyon",
  35: "Komedi",
  80: "Suç",
  99: "Belgesel",
  18: "Drama",
  10751: "Aile",
  14: "Fantastik",
  36: "Tarih",
  27: "Korku",
  10402: "Müzik",
  9648: "Gizem",
  10749: "Romantik",
  878: "Bilim Kurgu",
  53: "Gerilim",
  10752: "Savaş",
  37: "Kovboy",
};

function MovieDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const movie = location.state?.movie;
  const palette = Array.isArray(movie?.color_palette) ? movie.color_palette : [];
  const genres = Array.isArray(movie?.genre_ids)
    ? movie.genre_ids.map((genreId) => GENRE_MAP[genreId]).filter(Boolean)
    : [];

  const handleBackToSearch = () => {
    navigate("/search");
  };

  if (!movie) {
    return (
      <div style={styles.page}>
        <div style={styles.container}>
          <button type="button" style={styles.backButton} onClick={handleBackToSearch}>
            ← Önerilere Geri Dön
          </button>

          <div style={styles.emptyBox}>
            <h1 style={styles.emptyTitle}>Film detayı bulunamadı</h1>
            <p style={styles.emptyText}>
              Bu sayfa doğrudan açılmış olabilir. Lütfen arama sonuçlarından tekrar bir film seç.
            </p>
            <p style={styles.emptyId}>Film ID: {id}</p>
          </div>
        </div>
      </div>
    );
  }

  const formattedDate = movie.release_date
    ? new Date(movie.release_date).toLocaleDateString("tr-TR", {
        day: "numeric",
        month: "long",
        year: "numeric",
      })
    : movie.year || "Tarih bilgisi yok";

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <button type="button" style={styles.backButton} onClick={handleBackToSearch}>
          ← Önerilere Geri Dön
        </button>

        <section style={styles.heroCard}>
          <div style={styles.posterColumn}>
            {movie.poster_url ? (
              <img src={movie.poster_url} alt={movie.title} style={styles.poster} />
            ) : (
              <div style={styles.noPoster}>Poster Yok</div>
            )}
          </div>

          <div style={styles.infoColumn}>
            <div style={styles.topMeta}>
              <div>
                <h1 style={styles.title}>
                  {movie.title} {movie.year ? `(${movie.year})` : ""}
                </h1>

                <p style={styles.metaLine}>
                  {formattedDate}
                  {genres.length > 0 ? ` • ${genres.join(", ")}` : ""}
                </p>
              </div>

              {movie.vote_average && (
                <span style={styles.scoreBadge}>⭐ {movie.vote_average}</span>
              )}
            </div>

            <div style={styles.sectionBox}>
              <p style={styles.sectionLabel}>Özet</p>
              <p style={styles.sectionText}>
                {movie.overview || "Bu film için açıklama bilgisi bulunamadı."}
              </p>
            </div>

            <div style={styles.sectionBox}>
              <p style={styles.sectionLabel}>Neden önerildi?</p>
              <p style={styles.sectionText}>
                {movie.why_text || "Bu öneri için açıklama bilgisi bulunamadı."}
              </p>
            </div>

            <div style={styles.sectionBox}>
              <p style={styles.sectionLabel}>Duygu Eğrisi</p>
              <EmojiCurve curve={movie.emotion_curve} />
            </div>

            {palette.length > 0 && (
              <div style={styles.sectionBox}>
                <p style={styles.sectionLabel}>Renk Paleti</p>
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

            <div style={styles.sectionBox}>
              <p style={styles.sectionLabel}>Geri Bildirim</p>
              <FeedbackButtons filmId={movie.tmdb_id} />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

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
    padding: "32px 20px 48px",
  },
  container: {
    maxWidth: "1200px",
    margin: "0 auto",
  },
  backButton: {
    marginBottom: "24px",
    padding: "10px 18px",
    borderRadius: "999px",
    border: "1px solid rgba(255,255,255,0.15)",
    background: "rgba(15, 23, 42, 0.7)",
    color: "white",
    cursor: "pointer",
    fontWeight: "600",
  },
  heroCard: {
    display: "grid",
    gridTemplateColumns: "320px 1fr",
    gap: "28px",
    alignItems: "start",
    background: "linear-gradient(180deg, rgba(15,23,42,0.92) 0%, rgba(17,24,39,0.92) 100%)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "24px",
    padding: "24px",
    boxShadow: "0 24px 60px rgba(0,0,0,0.32)",
  },
  posterColumn: {
    width: "100%",
  },
  poster: {
    width: "100%",
    borderRadius: "20px",
    display: "block",
    objectFit: "cover",
    boxShadow: "0 16px 40px rgba(0,0,0,0.35)",
  },
  noPoster: {
    width: "100%",
    minHeight: "420px",
    borderRadius: "20px",
    background: "#111827",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#9ca3af",
    fontWeight: "700",
  },
  infoColumn: {
    minWidth: 0,
  },
  topMeta: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: "16px",
    marginBottom: "20px",
  },
  title: {
    margin: "0 0 10px 0",
    fontSize: "clamp(30px, 4vw, 46px)",
    lineHeight: "1.1",
    color: "#f8fafc",
  },
  metaLine: {
    margin: 0,
    color: "#cbd5e1",
    fontSize: "16px",
    lineHeight: "1.6",
  },
  scoreBadge: {
    flexShrink: 0,
    padding: "10px 12px",
    borderRadius: "999px",
    background: "rgba(251,191,36,0.14)",
    color: "#fcd34d",
    fontSize: "14px",
    fontWeight: "700",
    border: "1px solid rgba(251,191,36,0.18)",
  },
  sectionBox: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: "18px",
    padding: "16px",
    marginBottom: "16px",
  },
  sectionLabel: {
    margin: "0 0 8px 0",
    fontSize: "12px",
    fontWeight: "700",
    letterSpacing: "0.06em",
    textTransform: "uppercase",
    color: "#93c5fd",
  },
  sectionText: {
    margin: 0,
    color: "#dbeafe",
    fontSize: "15px",
    lineHeight: "1.75",
  },
  paletteRow: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
  },
  colorSwatch: {
    width: "34px",
    height: "34px",
    borderRadius: "999px",
    border: "1px solid rgba(255,255,255,0.2)",
  },
  emptyBox: {
    maxWidth: "800px",
    margin: "40px auto 0 auto",
    padding: "24px",
    borderRadius: "20px",
    background: "rgba(15, 23, 42, 0.8)",
    border: "1px solid rgba(255,255,255,0.08)",
    textAlign: "center",
  },
  emptyTitle: {
    marginTop: 0,
    marginBottom: "12px",
  },
  emptyText: {
    color: "#cbd5e1",
    lineHeight: "1.7",
  },
  emptyId: {
    color: "#94a3b8",
    marginTop: "14px",
    fontSize: "14px",
  },
};

export default MovieDetailPage;
