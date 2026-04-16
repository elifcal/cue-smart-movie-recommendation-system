import { useState } from "react";

function SearchBar({ onSearch, loading }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    const trimmedQuery = query.trim();
    if (!trimmedQuery || loading) return;

    onSearch(trimmedQuery);
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <div style={styles.inputRow}>
        <input
          type="text"
          placeholder="Örn: karanlık atmosferli gerilim filmi"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={styles.input}
          disabled={loading}
        />

        <button
          type="submit"
          disabled={loading}
          style={{
            ...styles.button,
            opacity: loading ? 0.8 : 1,
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Aranıyor..." : "Ara"}
        </button>
      </div>
    </form>
  );
}

const styles = {
  form: {
    width: "100%",
  },
  inputRow: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    gap: "12px",
    flexWrap: "wrap",
  },
  input: {
    flex: "1 1 620px",
    maxWidth: "620px",
    minWidth: "260px",
    padding: "15px 18px",
    borderRadius: "16px",
    border: "1px solid rgba(129, 140, 248, 0.22)",
    background: "rgba(255,255,255,0.94)",
    color: "#0f172a",
    fontSize: "15px",
    outline: "none",
    boxShadow:
      "0 0 0 1px rgba(255,255,255,0.04), 0 10px 30px rgba(0,0,0,0.08)",
  },
  button: {
    padding: "15px 22px",
    borderRadius: "16px",
    border: "1px solid rgba(99, 102, 241, 0.28)",
    background: "linear-gradient(135deg, #3b82f6 0%, #6366f1 100%)",
    color: "white",
    fontSize: "14px",
    fontWeight: "700",
    minWidth: "96px",
    boxShadow: "0 10px 24px rgba(59,130,246,0.22)",
  },
};

export default SearchBar;
