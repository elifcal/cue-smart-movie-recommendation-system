import { useState } from "react";
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL;
const DEFAULT_USER_ID = 1;

function CloseIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M18 6L6 18"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M6 6l12 12"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function FeedbackModal({
  open,
  onClose,
  movieId,
  movieTitle,
  userId = DEFAULT_USER_ID,
  onSuccess,
}) {
  const [submittingAction, setSubmittingAction] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  if (!open) return null;

  const handleFeedback = async (action) => {
    if (!movieId || !API_BASE_URL) {
      setErrorMessage("Geri bildirim gönderilemedi.");
      return;
    }

    setSubmittingAction(action);
    setErrorMessage("");

    try {
      await axios.post(`${API_BASE_URL}/feedback`, null, {
        params: {
          user_id: userId,
          film_id: movieId,
          action,
        },
      });

      onSuccess?.(action);
      onClose?.();
    } catch (error) {
      console.error("Feedback gönderme hatası:", error);
      setErrorMessage("Geri bildirim gönderilirken bir hata oluştu.");
    } finally {
      setSubmittingAction("");
    }
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <div>
            <h3 style={styles.title}>Geri Bildirim Ver</h3>
            <p style={styles.subtitle}>
              <strong style={styles.movieTitle}>
                {movieTitle || "Bu film"}
              </strong>{" "}
              hakkında ne düşünüyorsun?
            </p>
          </div>

          <button type="button" style={styles.closeButton} onClick={onClose}>
            <CloseIcon />
          </button>
        </div>

        <div style={styles.actions}>
          <button
            type="button"
            style={{
              ...styles.actionButton,
              ...styles.likeButton,
              ...(submittingAction ? styles.disabledButton : {}),
            }}
            onClick={() => handleFeedback("like")}
            disabled={Boolean(submittingAction)}
          >
            👍 Beğendim
          </button>

          <button
            type="button"
            style={{
              ...styles.actionButton,
              ...styles.dislikeButton,
              ...(submittingAction ? styles.disabledButton : {}),
            }}
            onClick={() => handleFeedback("dislike")}
            disabled={Boolean(submittingAction)}
          >
            👎 Beğenmedim
          </button>

          <button
            type="button"
            style={{
              ...styles.actionButton,
              ...styles.watchedButton,
              ...(submittingAction ? styles.disabledButton : {}),
            }}
            onClick={() => handleFeedback("watched")}
            disabled={Boolean(submittingAction)}
          >
            👁 İzledim
          </button>
        </div>

        {submittingAction ? (
          <p style={styles.infoText}>Geri bildirim gönderiliyor...</p>
        ) : null}

        {errorMessage ? <p style={styles.errorText}>{errorMessage}</p> : null}
      </div>
    </div>
  );
}

const styles = {
  overlay: {
    position: "fixed",
    inset: 0,
    zIndex: 200,
    background: "rgba(2, 6, 23, 0.72)",
    backdropFilter: "blur(8px)",
    WebkitBackdropFilter: "blur(8px)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "20px",
  },

  modal: {
    width: "100%",
    maxWidth: "520px",
    borderRadius: "24px",
    background:
      "linear-gradient(180deg, rgba(10,17,33,0.98) 0%, rgba(7,13,27,0.97) 100%)",
    border: "1px solid rgba(148,163,184,0.10)",
    boxShadow: "0 24px 48px rgba(0,0,0,0.35)",
    padding: "22px 22px 20px",
    color: "#f8fafc",
    boxSizing: "border-box",
  },

  header: {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: "16px",
    marginBottom: "18px",
  },

  title: {
    margin: "0 0 8px",
    fontSize: "1.5rem",
    fontWeight: "800",
    lineHeight: "1.2",
    letterSpacing: "-0.02em",
    color: "#ffffff",
  },

  subtitle: {
    margin: 0,
    color: "#cbd5e1",
    fontSize: "14px",
    lineHeight: "1.65",
  },

  movieTitle: {
    color: "#ffffff",
  },

  closeButton: {
    border: "none",
    background: "transparent",
    color: "#cbd5e1",
    cursor: "pointer",
    padding: "4px",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: "10px",
    flexShrink: 0,
  },

  actions: {
    display: "grid",
    gridTemplateColumns: "1fr",
    gap: "12px",
  },

  actionButton: {
    width: "100%",
    padding: "14px 16px",
    borderRadius: "16px",
    border: "1px solid transparent",
    fontSize: "15px",
    fontWeight: "800",
    cursor: "pointer",
    textAlign: "left",
    transition: "transform 0.15s ease, opacity 0.15s ease",
  },

  likeButton: {
    background: "rgba(34,197,94,0.12)",
    border: "1px solid rgba(34,197,94,0.18)",
    color: "#bbf7d0",
  },

  dislikeButton: {
    background: "rgba(248,113,113,0.12)",
    border: "1px solid rgba(248,113,113,0.18)",
    color: "#fecaca",
  },

  watchedButton: {
    background: "rgba(59,130,246,0.12)",
    border: "1px solid rgba(96,165,250,0.18)",
    color: "#bfdbfe",
  },

  disabledButton: {
    opacity: 0.72,
    cursor: "not-allowed",
  },

  infoText: {
    margin: "14px 0 0",
    color: "#cbd5e1",
    fontSize: "14px",
    fontWeight: "600",
  },

  errorText: {
    margin: "14px 0 0",
    color: "#fecaca",
    fontSize: "14px",
    fontWeight: "700",
  },
};

export default FeedbackModal;
