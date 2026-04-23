import { useEffect, useState } from "react";

function SearchIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M21 21l-4.35-4.35"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle
        cx="11"
        cy="11"
        r="6"
        stroke="currentColor"
        strokeWidth="2"
      />
    </svg>
  );
}

function SearchBar({ onSearch, loading = false, initialValue = "" }) {
  const [query, setQuery] = useState(initialValue);

  useEffect(() => {
    setQuery(initialValue || "");
  }, [initialValue]);

  const handleSubmit = (e) => {
    e.preventDefault();

    const trimmedQuery = query.trim();
    if (!trimmedQuery || loading) return;

    onSearch(trimmedQuery);
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <div style={styles.inputWrap}>
        <div style={styles.iconWrap}>
          <SearchIcon />
        </div>

        <input
          type="text"
          placeholder="Nasıl bir film arıyorsun?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={styles.input}
          disabled={loading}
        />
      </div>

      <button
        type="submit"
        style={{
          ...styles.button,
          ...(loading ? styles.buttonDisabled : {}),
        }}
        disabled={loading}
      >
        {loading ? "Aranıyor..." : "Ara"}
      </button>
    </form>
  );
}

const styles = {
  form: {
    display: "grid",
    gridTemplateColumns: "minmax(0, 1fr) auto",
    gap: "12px",
    width: "100%",
    alignItems: "center",
  },

  inputWrap: {
    display: "flex",
    alignItems: "center",
    minWidth: 0,
    minHeight: "72px",
    borderRadius: "999px",
    background: "rgba(248,250,252,0.96)",
    border: "1px solid rgba(148,163,184,0.16)",
    boxShadow:
      "inset 0 1px 0 rgba(255,255,255,0.35), 0 8px 20px rgba(15,23,42,0.10)",
    padding: "0 20px",
    boxSizing: "border-box",
  },

  iconWrap: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#64748b",
    marginRight: "12px",
    flexShrink: 0,
  },

  input: {
    flex: 1,
    minWidth: 0,
    border: "none",
    outline: "none",
    background: "transparent",
    color: "#0f172a",
    fontSize: "clamp(1rem, 1.4vw, 1.15rem)",
    fontWeight: "500",
    lineHeight: "1.4",
    padding: 0,
  },

  button: {
    minWidth: "128px",
    minHeight: "72px",
    padding: "0 24px",
    borderRadius: "999px",
    border: "1px solid rgba(99,102,241,0.24)",
    background: "linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%)",
    color: "#ffffff",
    fontSize: "15px",
    fontWeight: "800",
    cursor: "pointer",
    boxShadow: "0 12px 24px rgba(59,130,246,0.22)",
    transition: "transform 0.15s ease, opacity 0.15s ease",
    whiteSpace: "nowrap",
  },

  buttonDisabled: {
    cursor: "not-allowed",
    opacity: 0.78,
  },
};

export default SearchBar;
