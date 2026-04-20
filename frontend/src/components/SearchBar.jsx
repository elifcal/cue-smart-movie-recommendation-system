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
      <div style={styles.inputWrap}>
        <span style={styles.searchIcon}>⌕</span>
        <input
          type="text"
          placeholder="Nasıl bir film arıyorsun?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={styles.input}
          disabled={loading}
        />
      </div>

      <button type="submit" style={styles.button} disabled={loading}>
        {loading ? "Aranıyor..." : "Ara"}
      </button>
    </form>
  );
}

const styles = {
  form: {
    display: "grid",
    gridTemplateColumns: "1fr auto",
    gap: "16px",
    alignItems: "center",
    width: "100%",
  },
  inputWrap: {
    display: "flex",
    alignItems: "center",
    gap: "14px",
    height: "68px",
    padding: "0 22px",
    borderRadius: "999px",
    background: "#f8fafc",
    border: "1px solid rgba(148,163,184,0.20)",
    boxShadow: "0 10px 24px rgba(15,23,42,0.10)",
    boxSizing: "border-box",
  },
  searchIcon: {
    color: "#64748b",
    fontSize: "22px",
    lineHeight: 1,
    flexShrink: 0,
  },
  input: {
    flex: 1,
    border: "none",
    outline: "none",
    background: "transparent",
    fontSize: "18px",
    fontWeight: "500",
    color: "#0f172a",
    lineHeight: "1.5",
  },
  button: {
    height: "68px",
    padding: "0 28px",
    border: "none",
    borderRadius: "999px",
    background: "linear-gradient(90deg, #6366f1 0%, #3b82f6 100%)",
    color: "white",
    fontSize: "17px",
    fontWeight: "800",
    cursor: "pointer",
    boxShadow: "0 12px 28px rgba(59,130,246,0.22)",
    whiteSpace: "nowrap",
  },
};

export default SearchBar;
