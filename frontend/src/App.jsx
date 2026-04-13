import { useEffect, useState } from "react";
import axios from "axios";
import SearchBar from "./components/SearchBar";
import MovieCard from "./components/MovieCard";

const API_BASE_URL = "http://127.0.0.1:8000";
const TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500";

function App() {
  const [healthStatus, setHealthStatus] = useState("Kontrol ediliyor...");
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`);
        setHealthStatus(response.data.status || "ok");
      } catch {
        setHealthStatus("Backend'e bağlanılamadı");
      }
    };

    checkHealth();
  }, []);

  const handleSearch = async (query) => {
    if (!query.trim()) return;

    setLoading(true);

    try {
      const response = await axios.get(`${API_BASE_URL}/search`, {
        params: { q: query },
      });

      const formattedMovies = response.data.results.map((movie) => ({
        ...movie,
        tmdb_id: movie.id,
        poster_url: movie.poster_path
          ? `${TMDB_IMAGE_BASE_URL}${movie.poster_path}`
          : "",
      }));

      setMovies(formattedMovies);
    } catch (error) {
      console.error("Search error:", error);
      setMovies([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.title}>Cue Smart Movie Recommendation</h1>
      <p style={styles.status}>API Durumu: {healthStatus}</p>

      <SearchBar onSearch={handleSearch} />

      {loading && <p style={styles.message}>Filmler yükleniyor...</p>}

      {!loading && movies.length === 0 && (
        <p style={styles.message}>
          Arama yaparak backend’den gerçek sonuçları getir.
        </p>
      )}

      <div style={styles.grid}>
        {movies.map((movie) => (
          <MovieCard key={movie.tmdb_id} movie={movie} />
        ))}
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#111827",
    color: "white",
    padding: "32px",
  },
  title: {
    textAlign: "center",
    marginBottom: "12px",
  },
  status: {
    textAlign: "center",
    marginBottom: "24px",
    color: "#d1d5db",
  },
  message: {
    textAlign: "center",
    color: "#d1d5db",
    marginTop: "20px",
    marginBottom: "20px",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
    gap: "20px",
    marginTop: "24px",
  },
};

export default App;
