import { useState } from "react";
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL;

function FeedbackButtons({ filmId }) {
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedAction, setSelectedAction] = useState("");

  const sendFeedback = async (action) => {
    if (!filmId) return;

    setLoading(true);
    setStatus("");

    try {
      await axios.post(`${API_BASE_URL}/feedback`, null, {
        params: {
          film_id: filmId,
          action,
        },
      });

      setSelectedAction(action);

      if (action === "like") {
        setStatus("Beğeni gönderildi");
      } else if (action === "dislike") {
        setStatus("Beğenmeme gönderildi");
      } else if (action === "watched") {
        setStatus("İzlendi olarak işaretlendi");
      }
    } catch (error) {
      console.error("Feedback error:", error);
      setStatus("Geri bildirim gönderilemedi");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.wrapper}>
      <p style={styles.label}>Geri Bildirim</p>

      <div style={styles.row}>
        <button
          style={{
            ...styles.button,
            ...styles.likeButton,
            ...(selectedAction === "like" ? styles.selectedButton : {}),
          }}
          onClick={() => sendFeedback("like")}
          disabled={loading}
        >
          👍 Beğendim
        </button>

        <button
          style={{
            ...styles.button,
            ...styles.dislikeButton,
            ...(selectedAction === "dislike" ? styles.selectedButton : {}),
          }}
          onClick={() => sendFeedback("dislike")}
          disabled={loading}
        >
          👎 Beğenmedim
        </button>

        <button
          style={{
            ...styles.button,
            ...styles.watchedButton,
            ...(selectedAction === "watched" ? styles.selectedButton : {}),
          }}
          onClick={() => sendFeedback("watched")}
          disabled={loading}
        >
          👀 Zaten İzledim
        </button>
      </div>

      {status && <p style={styles.status}>{status}</p>}
    </div>
  );
}

const styles = {
  wrapper: {
    marginTop: "16px",
  },
  label: {
    fontSize: "13px",
    color: "#d1d5db",
    marginBottom: "8px",
  },
  row: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
  },
  button: {
    padding: "10px 14px",
    borderRadius: "10px",
    border: "1px solid transparent",
    cursor: "pointer",
    color: "white",
    fontWeight: "600",
    transition: "all 0.2s ease",
  },
  likeButton: {
    background: "#16a34a",
  },
  dislikeButton: {
    background: "#dc2626",
  },
  watchedButton: {
    background: "#2563eb",
  },
  selectedButton: {
    border: "1px solid rgba(255,255,255,0.8)",
    boxShadow: "0 0 0 3px rgba(255,255,255,0.12)",
    transform: "translateY(-1px)",
  },
  status: {
    marginTop: "10px",
    fontSize: "13px",
    color: "#d1d5db",
  },
};

export default FeedbackButtons;
