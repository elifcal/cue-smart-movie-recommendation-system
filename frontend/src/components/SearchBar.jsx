import { useState } from "react";

function SearchBar({ onSearch }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
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
  );
}

const styles = {
  form: {
    display: "flex",
    gap: "12px",
    justifyContent: "center",
    marginBottom: "24px",
    flexWrap: "wrap"
  },
  input: {
    width: "420px",
    maxWidth: "100%",
    padding: "12px",
    borderRadius: "10px",
    border: "1px solid #374151",
    outline: "none"
  },
  button: {
    padding: "12px 20px",
    borderRadius: "10px",
    border: "none",
    background: "#2563eb",
    color: "white",
    cursor: "pointer"
  }
};

export default SearchBar;
