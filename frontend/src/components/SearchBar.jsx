import { useState } from "react";

const exampleQueries = [
  "90'larda geçen korku filmi",
  "psikolojik gerilim",
  "twist sonlu film",
];

function SearchBar({ onSearch }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(query);
  };

  const handleExampleClick = (example) => {
    setQuery(example);
    onSearch(example);
  };

  return (
    <div style={styles.wrapper}>
      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          type="text"
          placeholder="Örn: karanlık atmosferli gerilim filmi"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={styles.input}
        />
        <button type="submit" style={styles.button}>
          Ara
        </button>
      </form>

      <div style={styles.examplesWrapper}>
        <p style={styles.examplesLabel}>Örnek aramalar</p>
        <div style={styles.examplesRow}>
          {exampleQueries.map((example) => (
            <button
              key={example}
              type="button"
              style={styles.exampleButton}
              onClick={() => handleExampleClick(example)}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

const styles = {
  wrapper: {
    marginBottom: "24px",
  },
  form: {
    display: "flex",
    gap: "12px",
    justifyContent: "center",
    flexWrap: "wrap",
  },
  input: {
    width: "420px",
    maxWidth: "100%",
    padding: "12px",
    borderRadius: "10px",
    border: "1px solid #374151",
    outline: "none",
  },
  button: {
    padding: "12px 20px",
    borderRadius: "10px",
    border: "none",
    background: "#2563eb",
    color: "white",
    cursor: "pointer",
  },
  examplesWrapper: {
    marginTop: "16px",
    textAlign: "center",
  },
  examplesLabel: {
    color: "#d1d5db",
    fontSize: "14px",
    marginBottom: "10px",
  },
  examplesRow: {
    display: "flex",
    justifyContent: "center",
    gap: "10px",
    flexWrap: "wrap",
  },
  exampleButton: {
    padding: "8px 12px",
    borderRadius: "999px",
    border: "1px solid #374151",
    background: "#1f2937",
    color: "white",
    cursor: "pointer",
    fontSize: "13px",
  },
};

export default SearchBar;
