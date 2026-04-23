function EmojiCurve({ curve = [] }) {
  if (!Array.isArray(curve) || curve.length === 0) {
    return null;
  }

  const getSafeValue = (value) => {
    const num = Number(value);
    if (Number.isNaN(num)) return 0;
    return Math.max(0, Math.min(num, 1));
  };

  return (
    <div style={styles.wrapper}>
      <div style={styles.header}>
        <p style={styles.label}>Duygu Eğrisi</p>
        <span style={styles.badge}>{curve.length} sahne</span>
      </div>

      <div style={styles.chart}>
        {curve.map((value, index) => {
          const safeValue = getSafeValue(value);
          const percent = Math.max(10, safeValue * 100);

          return (
            <div key={index} style={styles.barGroup}>
              <div style={styles.valueLabel}>{safeValue.toFixed(2)}</div>

              <div style={styles.barTrack}>
                <div
                  style={{
                    ...styles.barFill,
                    height: `${percent}%`,
                  }}
                  title={`Sahne ${index + 1}: ${safeValue.toFixed(2)}`}
                />
              </div>

              <div style={styles.sceneLabel}>{index + 1}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const styles = {
  wrapper: {
    marginTop: "18px",
    padding: "16px",
    borderRadius: "16px",
    background: "rgba(17, 24, 39, 0.92)",
    border: "1px solid rgba(255,255,255,0.08)",
    boxShadow: "0 10px 30px rgba(0,0,0,0.22)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
    marginBottom: "14px",
  },
  label: {
    margin: 0,
    fontSize: "14px",
    fontWeight: 700,
    color: "#f9fafb",
    letterSpacing: "0.2px",
  },
  badge: {
    fontSize: "12px",
    color: "#cbd5e1",
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "999px",
    padding: "4px 10px",
    whiteSpace: "nowrap",
  },
  chart: {
    minHeight: "200px",
    display: "flex",
    alignItems: "flex-end",
    gap: "10px",
    overflowX: "auto",
    paddingBottom: "4px",
  },
  barGroup: {
    minWidth: "34px",
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "8px",
  },
  valueLabel: {
    fontSize: "11px",
    color: "#93c5fd",
    lineHeight: 1,
    minHeight: "14px",
  },
  barTrack: {
    position: "relative",
    width: "100%",
    maxWidth: "22px",
    height: "120px",
    display: "flex",
    alignItems: "flex-end",
    justifyContent: "center",
    borderRadius: "999px",
    background: "rgba(255,255,255,0.07)",
    overflow: "hidden",
    border: "1px solid rgba(255,255,255,0.06)",
  },
  barFill: {
    width: "100%",
    borderRadius: "999px",
    background: "linear-gradient(to top, #60a5fa 0%, #818cf8 50%, #c084fc 100%)",
    boxShadow: "0 0 18px rgba(129, 140, 248, 0.35)",
    transition: "height 0.35s ease",
  },
  sceneLabel: {
    fontSize: "11px",
    color: "#9ca3af",
    lineHeight: 1,
  },
};

export default EmojiCurve;
