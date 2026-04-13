function EmojiCurve({ curve = [] }) {
  if (!Array.isArray(curve) || curve.length === 0) {
    return null;
  }

  return (
    <div style={styles.wrapper}>
      <p style={styles.label}>Duygu Eğrisi</p>

      <div style={styles.chart}>
        {curve.map((value, index) => {
          const safeValue = Math.max(0, Math.min(value, 1));
          const height = Math.max(16, safeValue * 300);

          return (
            <div key={index} style={styles.barWrapper}>
              <div
                style={{
                  ...styles.bar,
                  height: `${height}px`,
                }}
                title={`Sahne ${index + 1}: ${safeValue.toFixed(2)}`}
              />
            </div>
          );
        })}
      </div>
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
  chart: {
    height: "140px",
    display: "flex",
    alignItems: "flex-end",
    gap: "6px",
    padding: "10px",
    background: "#111827",
    borderRadius: "12px",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  barWrapper: {
    flex: 1,
    display: "flex",
    alignItems: "flex-end",
    justifyContent: "center",
    height: "100%",
  },
  bar: {
    width: "100%",
    maxWidth: "18px",
    borderRadius: "999px",
    background: "linear-gradient(to top, #60a5fa, #c084fc)",
  },
};

export default EmojiCurve;
