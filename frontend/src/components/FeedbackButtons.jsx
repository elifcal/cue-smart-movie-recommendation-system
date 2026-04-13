import { useState } from "react";
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL;

function FeedbackButtons({ filmId }) {
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

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

      setStatus(action === "like" ? "Beğeni gönderildi" : "Beğenmeme gönderildi");
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
          style={styles.likeButton}
          onClick={() => sendFeedback("like")}
          disabled={loading}
        >
          👍 Beğendim
        </button>

        <button
          style={styles.dislikeButton}
          onClick={() => sendFeedback("dislike")}
          disabled={loading}
        >
          👎 Beğenmedim
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
  likeButton: {
    padding: "10px 14px",
    borderRadius: "10px",
    border: "none",
    cursor: "pointer",
    background: "#16a34a",
    color: "white",
    fontWeight: "600",
  },
  dislikeButton: {
    padding: "10px 14px",
    borderRadius: "10px",
    border: "none",
    cursor: "pointer",
    background: "#dc2626",
    color: "white",
    fontWeight: "600",
  },
  status: {
    marginTop: "10px",
    fontSize: "13px",
    color: "#d1d5db",
  },
};

export default FeedbackButtons;
