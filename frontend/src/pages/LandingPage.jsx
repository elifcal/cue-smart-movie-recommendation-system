import { useNavigate } from "react-router-dom";

function LandingPage() {
  const navigate = useNavigate();

  return (
    <div style={styles.page}>
      <div style={styles.overlay} />

      <div style={styles.container}>
        <div style={styles.badge}>Akıllı Film Öneri Sistemi</div>

        <h1 style={styles.title}>
          <span style={styles.titleCue}>Cue</span>
          <span style={styles.titleRest}>Smart Movie Recommendation</span>
        </h1>

        <p style={styles.subtitle}>
          Nasıl bir film aradığını doğal dilde anlat, sana ruh haline ve
          beklentine uygun film önerileri sunalım.
        </p>

        <button
          type="button"
          style={styles.button}
          onClick={() => navigate("/search")}
        >
          Film Aramaya Başla
        </button>
      </div>
    </div>
  );
}

const styles = {
  page: {
    position: "relative",
    minHeight: "100vh",
    width: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "flex-start",
    padding: "clamp(24px, 4vw, 56px)",
    boxSizing: "border-box",
    overflow: "hidden",
    backgroundImage: "url('/landing-bg.jpg')",
    backgroundSize: "cover",
    backgroundPosition: "center center",
    backgroundRepeat: "no-repeat",
    color: "white",
  },
  overlay: {
    position: "absolute",
    inset: 0,
    background: `
      linear-gradient(
        90deg,
        rgba(3, 2, 20, 0.92) 0%,
        rgba(7, 4, 32, 0.82) 28%,
        rgba(18, 8, 50, 0.62) 52%,
        rgba(28, 10, 64, 0.38) 72%,
        rgba(22, 8, 48, 0.26) 100%
      )
    `,
    zIndex: 1,
  },
  container: {
    position: "relative",
    zIndex: 2,
    width: "100%",
    maxWidth: "760px",
    textAlign: "left",
    padding: "clamp(20px, 3vw, 36px)",
    borderRadius: "28px",
    backdropFilter: "blur(6px)",
    background: "rgba(8, 6, 30, 0.18)",
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.35)",
  },
  badge: {
    display: "inline-block",
    marginBottom: "18px",
    padding: "9px 16px",
    borderRadius: "999px",
    background: "rgba(139, 92, 246, 0.18)",
    color: "#ddd6fe",
    fontSize: "13px",
    fontWeight: "700",
    border: "1px solid rgba(196, 181, 253, 0.25)",
    boxShadow: "0 0 24px rgba(139, 92, 246, 0.16)",
  },
  title: {
    margin: "0 0 18px 0",
  },
  titleCue: {
    display: "block",
    fontSize: "clamp(64px, 11vw, 124px)",
    lineHeight: "0.9",
    fontWeight: "800",
    letterSpacing: "-0.08em",
    background: "linear-gradient(90deg, #ffffff 0%, #eadcff 45%, #c4b5fd 100%)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    textShadow: "0 0 36px rgba(168, 85, 247, 0.22)",
  },
  titleRest: {
    display: "block",
    marginTop: "8px",
    fontSize: "clamp(24px, 3.2vw, 46px)",
    lineHeight: "1.08",
    fontWeight: "700",
    color: "#f8fafc",
    letterSpacing: "-0.03em",
  },
  subtitle: {
    maxWidth: "680px",
    margin: "0 0 32px 0",
    color: "rgba(226, 232, 240, 0.92)",
    fontSize: "clamp(16px, 1.4vw, 19px)",
    lineHeight: "1.8",
  },
  button: {
    padding: "15px 30px",
    borderRadius: "999px",
    border: "none",
    cursor: "pointer",
    fontSize: "16px",
    fontWeight: "700",
    background: "linear-gradient(90deg, #8b5cf6 0%, #3b82f6 100%)",
    color: "white",
    boxShadow: "0 12px 30px rgba(99, 102, 241, 0.35)",
    transition: "transform 0.2s ease, box-shadow 0.2s ease",
  },
};

export default LandingPage;
