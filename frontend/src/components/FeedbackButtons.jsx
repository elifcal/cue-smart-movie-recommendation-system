import { useState } from "react";
import axios from "axios";
import { trackEvent } from "../utils/analytics";

const API_BASE_URL = import.meta.env.VITE_API_URL;

function FeedbackButtons({ filmId, genre, rating }) {
  const [status, setStatus] = useState("");
  const [statusType, setStatusType] = useState("info");
  const [loadingAction, setLoadingAction] = useState("");
  const [selectedAction, setSelectedAction] = useState("");

  const actionConfig = {
    like: {
      label: "Beğendim",
      icon: "👍",
      successText: "Tercihin kaydedildi",
    },
    dislike: {
      label: "Bana Uygun Değil",
      icon: "👎",
      successText: "Bu geri bildirim kaydedildi",
    },
    watched: {
      label: "Zaten İzledim",
      icon: "👀",
      successText: "Film izlendi olarak işaretlendi",
    },
  };

  const sendFeedback = async (action) => {
    if (!filmId) return;
    if (loadingAction) return;
    if (selectedAction === action) return;

    setLoadingAction(action);
    setStatus("");

    try {
      await axios.post(`${API_BASE_URL}/feedback`, null, {
        params: {
          film_id: filmId,
          action,
        },
      });

      setSelectedAction(action);
      setStatus(actionConfig[action].successText);
      setStatusType("success");

      const eventParams = {
        film_id: String(filmId),
        feedback_type: action,
        source: "movie_detail",
      };

      if (genre) {
        eventParams.genre_primary = genre;
      }

      if (typeof rating === "number") {
        eventParams.movie_rating = Number(rating.toFixed(1));
      }

      trackEvent("feedback_submit", eventParams);
    } catch (error) {
      console.error("Feedback error:", error);
      setStatus("Geri bildirim gönderilemedi");
      setStatusType("error");

      trackEvent("feedback_submit_error", {
        film_id: String(filmId),
        feedback_type: action,
        source: "movie_detail",
      });
    } finally {
      setLoadingAction("");
    }
  };

  const getButtonStyle = (action) => {
    const isSelected = selectedAction === action;
    const isDisabled = Boolean(loadingAction);

    if (action === "like") {
      return {
        ...styles.button,
        ...styles.likeButton,
        ...(isSelected ? styles.likeSelected : {}),
        ...(isDisabled ? styles.buttonDisabled : {}),
      };
    }

    if (action === "dislike") {
      return {
        ...styles.button,
        ...styles.dislikeButton,
        ...(isSelected ? styles.dislikeSelected : {}),
        ...(isDisabled ? styles.buttonDisabled : {}),
      };
    }

    return {
      ...styles.button,
      ...styles.watchedButton,
      ...(isSelected ? styles.watchedSelected : {}),
      ...(isDisabled ? styles.buttonDisabled : {}),
    };
  };

  const getStatusStyle = () => {
    if (statusType === "success") {
      return {
        ...styles.status,
        ...styles.statusSuccess,
      };
    }

    if (statusType === "error") {
      return {
        ...styles.status,
        ...styles.statusError,
      };
    }

    return styles.status;
  };

  return (
    <div style={styles.wrapper}>
      <div style={styles.header}>
        <p style={styles.label}>Geri Bildirim</p>
        <span style={styles.helperText}>
          Bu veriler önerileri geliştirmek için kullanılır
        </span>
      </div>

      <div style={styles.row}>
        {Object.entries(actionConfig).map(([action, config]) => {
          const isLoading = loadingAction === action;

          return (
            <button
              key={action}
              type="button"
              aria-pressed={selectedAction === action}
              onClick={() => sendFeedback(action)}
              disabled={Boolean(loadingAction)}
              style={getButtonStyle(action)}
            >
              <span style={styles.icon}>{config.icon}</span>
              <span>{isLoading ? "Gönderiliyor..." : config.label}</span>
            </button>
          );
        })}
      </div>

      {status && <div style={getStatusStyle()}>{status}</div>}
    </div>
  );
}

const styles = {
  wrapper: {
    marginTop: "18px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "12px",
    marginBottom: "10px",
    flexWrap: "wrap",
  },
  label: {
    margin: 0,
    fontSize: "13px",
    fontWeight: "700",
    color: "#e5e7eb",
    letterSpacing: "0.02em",
  },
  helperText: {
    fontSize: "12px",
    color: "rgba(209,213,219,0.78)",
  },
  row: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
    gap: "10px",
  },
  button: {
    minHeight: "48px",
    padding: "12px 14px",
    borderRadius: "14px",
    border: "1px solid rgba(255,255,255,0.10)",
    cursor: "pointer",
    color: "#ffffff",
    fontWeight: "700",
    fontSize: "14px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "8px",
    transition: "all 0.2s ease",
    boxShadow: "0 10px 24px rgba(0,0,0,0.18)",
    backdropFilter: "blur(10px)",
  },
  icon: {
    fontSize: "15px",
    lineHeight: 1,
  },
  likeButton: {
    background:
      "linear-gradient(135deg, rgba(22,163,74,0.96), rgba(34,197,94,0.82))",
  },
  dislikeButton: {
    background:
      "linear-gradient(135deg, rgba(220,38,38,0.96), rgba(239,68,68,0.82))",
  },
  watchedButton: {
    background:
      "linear-gradient(135deg, rgba(37,99,235,0.96), rgba(59,130,246,0.82))",
  },
  likeSelected: {
    border: "1px solid rgba(255,255,255,0.38)",
    boxShadow: "0 0 0 3px rgba(34,197,94,0.18)",
    transform: "translateY(-1px)",
  },
  dislikeSelected: {
    border: "1px solid rgba(255,255,255,0.38)",
    boxShadow: "0 0 0 3px rgba(239,68,68,0.18)",
    transform: "translateY(-1px)",
  },
  watchedSelected: {
    border: "1px solid rgba(255,255,255,0.38)",
    boxShadow: "0 0 0 3px rgba(59,130,246,0.18)",
    transform: "translateY(-1px)",
  },
  buttonDisabled: {
    opacity: 0.74,
    cursor: "not-allowed",
  },
  status: {
    marginTop: "12px",
    padding: "10px 12px",
    borderRadius: "12px",
    fontSize: "13px",
    fontWeight: "600",
    border: "1px solid rgba(255,255,255,0.08)",
    background: "rgba(255,255,255,0.06)",
    color: "#e5e7eb",
  },
  statusSuccess: {
    background: "rgba(34,197,94,0.10)",
    color: "#bbf7d0",
    border: "1px solid rgba(34,197,94,0.20)",
  },
  statusError: {
    background: "rgba(239,68,68,0.10)",
    color: "#fecaca",
    border: "1px solid rgba(239,68,68,0.20)",
  },
};

export default FeedbackButtons;
