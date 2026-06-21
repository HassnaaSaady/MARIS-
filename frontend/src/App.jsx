/**
 * App.jsx — Maritime Navigation AI System
 * Complete React dashboard with all 10 features.
 * Uses Leaflet for maps, Recharts for charts.
 */
import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  MapContainer, TileLayer, Marker, Popup,
  CircleMarker, Polyline, useMap,
} from "react-leaflet";
import L from "leaflet";
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import "leaflet/dist/leaflet.css";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";
const WS  = process.env.REACT_APP_WS_URL  || "ws://localhost:8000/ws/vessels";

const RISK_COLOR = {
  HIGH: "#EF5350", MEDIUM: "#FFA726", LOW: "#66BB6A", CRITICAL: "#B71C1C",
};
const CHART_COLORS = [
  "#42A5F5","#FF7043","#AB47BC","#26A69A",
  "#FFA726","#8D6E63","#26C6DA","#D4E157",
];

// API stores vessel_type as AIS numeric codes like "70.0", not labels like "Cargo".
const VESSEL_TYPE_OPTIONS = [
  { value: "70.0", label: "Cargo" },
  { value: "80.0", label: "Tanker" },
  { value: "60.0", label: "Passenger" },
  { value: "30.0", label: "Fishing" },
  { value: "37.0", label: "Pleasure Craft" },
  { value: "52.0", label: "Towing / Tug" },
  { value: "35.0", label: "Military" },
  { value: "36.0", label: "Sailing" },
];

// Vessel icon rotated by heading
const makeVesselIcon = (heading, color) => {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 22 22">
      <g transform="rotate(${heading || 0}, 11, 11)">
        <polygon points="11,2 19,20 11,15 3,20"
          fill="${color || '#90A4AE'}"
          stroke="white" stroke-width="1.5"/>
      </g>
    </svg>`;
  return L.divIcon({
    html: svg, className: "",
    iconSize: [22, 22], iconAnchor: [11, 11],
  });
};

// ── API helpers ────────────────────────────────────────────────────────────────
async function apiFetch(path, params = {}) {
  try {
    const url = new URL(`${API}${path}`);

    Object.entries(params).forEach(([k, v]) => {
      if (
        v !== undefined &&
        v !== null &&
        v !== "" &&
        v !== "All" &&
        v !== "All Types"
      ) {
        url.searchParams.set(k, v);
      }
    });

    console.log("Fetching:", url.toString());

    const res = await fetch(url);

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    return await res.json();
  } catch (e) {
    console.error(`API error [${path}]:`, e);
    return null;
  }
}

// ── Custom hooks ───────────────────────────────────────────────────────────────
function useVessels(filters) {
  const [vessels, setVessels] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetch_ = useCallback(async () => {
    const data = await apiFetch("/api/vessels", filters);

    if (data) {
      const parsedVessels = data.vessels || [];
      console.log("Vessels response count:", data.count);
      console.log("Parsed vessels length:", parsedVessels.length);
      setVessels(parsedVessels);
    }

    setLoading(false);
  }, [JSON.stringify(filters)]);

  useEffect(() => {
    fetch_();
    const t = setInterval(fetch_, 5000);
    return () => clearInterval(t);
  }, [fetch_]);

  return { vessels, loading, refetch: fetch_ };
}


// Polls every 3s, shows last 1 hour of alerts only
function useAlerts() {
  const [alerts, setAlerts] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(null);

  const load = useCallback(async () => {
    const d = await apiFetch("/api/alerts", { hours_back: 1, limit: 100 });
    if (d) {
      setAlerts(d.alerts || []);
      setLastUpdate(new Date());
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [load]);

  return { alerts, lastUpdate };
}


// Polls analytics summary for Analytics panel
function useAnalytics() {
  const [data, setData] = useState(null);

  const load = useCallback(async () => {
    const d = await apiFetch("/api/analytics/summary", { days_back: 14 });
    if (d) setData(d);
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, [load]);

  return data;
}

// ── CriticalBanner ─────────────────────────────────────────────────────────────
const CriticalBanner = ({ alert, onDismiss }) => {
  if (!alert) return null;
  return (
    <div style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 9999,
      background: "#7f1d1d", borderBottom: "3px solid #EF5350",
      padding: "12px 24px", display: "flex", alignItems: "center",
      justifyContent: "space-between", boxShadow: "0 4px 20px rgba(239,83,80,0.4)",
    }}>
      <span style={{ color: "#fff", fontWeight: 700, fontSize: 15 }}>
        ⚠️ COLLISION RISK DETECTED —{" "}
        <span style={{ color: "#fca5a5" }}>
          {alert.vessel_1 || alert.mmsi_1}
        </span>{" "}
        and{" "}
        <span style={{ color: "#fca5a5" }}>
          {alert.vessel_2 || alert.mmsi_2}
        </span>
        {alert.distance_nm != null && (
          <span style={{ fontWeight: 400, fontSize: 13, color: "#fca5a5", marginLeft: 8 }}>
            ({(alert.distance_nm).toFixed(3)} nm apart)
          </span>
        )}
      </span>
      <button onClick={onDismiss} style={{
        background: "none", border: "1px solid #fca5a5",
        color: "#fca5a5", borderRadius: 4, padding: "4px 12px",
        cursor: "pointer", fontSize: 13,
      }}>
        ✕ Dismiss
      </button>
    </div>
  );
};

// ── Haversine distance (nautical miles) ───────────────────────────────────────
const haversineNm = (lat1, lon1, lat2, lon2) => {
  const R = 3440.065;
  const toRad = d => d * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};

// ── Dead-reckoning: predict position N minutes ahead ─────────────────────────
const predictPosition = (lat, lon, sogKn, headingDeg, minutes = 5) => {
  if (!sogKn || sogKn < 0.1) return null;
  const distNm = sogKn * (minutes / 60);
  const hdRad = headingDeg * Math.PI / 180;
  const newLat = lat + (distNm * Math.cos(hdRad)) / 60;
  const newLon = lon + (distNm * Math.sin(hdRad)) / (60 * Math.cos(lat * Math.PI / 180));
  return { lat: newLat, lon: newLon };
};

// ── UI Components ──────────────────────────────────────────────────────────────
const MetricCard = ({ label, value, color = "#3B82F6", icon, sub }) => (
  <div style={{
    background: "#1C2333", borderRadius: 10, padding: "14px 18px",
    border: "1px solid #2D3748", display: "flex", alignItems: "center", gap: 14,
  }}>
    <span style={{ fontSize: 28 }}>{icon}</span>
    <div>
      <p style={{ color: "#A0AEC0", fontSize: 12, margin: 0 }}>{label}</p>
      <p style={{ color, fontSize: 22, fontWeight: 700, margin: 0 }}>{value ?? "—"}</p>
      {sub && <p style={{ color: "#718096", fontSize: 11, margin: 0 }}>{sub}</p>}
    </div>
  </div>
);

const Badge = ({ severity }) => {
  const colors = {
    CRITICAL: { bg: "#7f1d1d", text: "#fca5a5" },
    HIGH:     { bg: "#7c2d12", text: "#fdba74" },
    MEDIUM:   { bg: "#713f12", text: "#fde68a" },
    LOW:      { bg: "#14532d", text: "#86efac" },
  };
  const c = colors[severity] || colors.LOW;
  return (
    <span style={{
      background: c.bg, color: c.text,
      padding: "2px 8px", borderRadius: 4,
      fontSize: 11, fontWeight: 700,
    }}>
      {severity}
    </span>
  );
};

const VesselPopup = ({ v }) => {
  const pred = predictPosition(v.lat, v.lon, v.sog, v.heading || v.cog || 0);
  const distNm = pred ? haversineNm(v.lat, v.lon, pred.lat, pred.lon) : null;
  return (
    <div style={{ minWidth: 210, fontSize: 13 }}>
      <b style={{ fontSize: 15 }}>{v.vessel_name || v.mmsi}</b>
      <table style={{ width: "100%", marginTop: 6, fontSize: 12 }}>
        <tbody>
          {[
            ["MMSI",    v.mmsi],
            ["Type", v.vessel_type_label],
            ["Speed",   `${(v.sog||0).toFixed(1)} kn`],
            ["Heading", `${(v.heading||0).toFixed(0)}°`],
            ["Risk",    v.risk_level],
            ["Anomaly", v.is_anomaly ? `Yes (${v.anomaly_type})` : "No"],
            ["Updated", v.updated_at?.slice(0,19)],
          ].map(([k, val]) => (
            <tr key={k}>
              <td style={{ color: "#666", paddingRight: 8, fontWeight: 600 }}>{k}</td>
              <td>{val || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {pred ? (
        <div style={{ marginTop: 8, borderTop: "1px solid #ddd", paddingTop: 6 }}>
          <b style={{ fontSize: 12, color: "#555" }}>Predicted Position (5 min)</b>
          <table style={{ width: "100%", marginTop: 4, fontSize: 12 }}>
            <tbody>
              {[
                ["Lat",      pred.lat.toFixed(5) + "°"],
                ["Lon",      pred.lon.toFixed(5) + "°"],
                ["Distance", distNm?.toFixed(3) + " nm"],
              ].map(([k, val]) => (
                <tr key={k}>
                  <td style={{ color: "#666", paddingRight: 8, fontWeight: 600 }}>{k}</td>
                  <td style={{ color: "#42A5F5" }}>{val}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ marginTop: 8, borderTop: "1px solid #ddd", paddingTop: 6,
                      fontSize: 11, color: "#999" }}>
          Predicted Position (5 min): N/A (vessel stationary)
        </div>
      )}
    </div>
  );
};

// ── Sidebar ────────────────────────────────────────────────────────────────────
const PAGES = [
  { id: "map",      label: "Vessel Map",       icon: "🗺️" },
  { id: "replay",   label: "Historical Replay", icon: "⏮️" },
  { id: "heatmap",  label: "Traffic Heatmap",   icon: "🔥" },
  { id: "anomaly",  label: "Anomaly Detection", icon: "⚠️" },
  { id: "collision",label: "Collision Risk",     icon: "🚨" },
  { id: "alerts",   label: "Alerts",             icon: "🔔" },
  { id: "analytics",label: "Analytics",          icon: "📊" },
  { id: "search",   label: "Search",             icon: "🔍" },
];

const Sidebar = ({ active, onChange, alertCount }) => (
  <aside style={{
    width: 220, background: "#0D1117",
    borderRight: "1px solid #21262D",
    display: "flex", flexDirection: "column",
    padding: "20px 8px", gap: 4,
  }}>
    <div style={{
      color: "#58A6FF", fontWeight: 700, fontSize: 16,
      padding: "0 8px 16px",
    }}>
      🚢 Maritime AI
    </div>
    {PAGES.map(p => (
      <button key={p.id} onClick={() => onChange(p.id)}
        style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 12px", borderRadius: 8,
          background: active === p.id ? "#1F6FEB22" : "transparent",
          color: active === p.id ? "#58A6FF" : "#8B949E",
          border: active === p.id ? "1px solid #1F6FEB44" : "1px solid transparent",
          cursor: "pointer", fontSize: 14, width: "100%", textAlign: "left",
          fontWeight: active === p.id ? 600 : 400,
          transition: "all 0.15s",
        }}>
        <span>{p.icon}</span>
        <span style={{ flex: 1 }}>{p.label}</span>
        {p.id === "alerts" && alertCount > 0 && (
          <span style={{
            background: "#EF5350", color: "#fff",
            borderRadius: "50%", minWidth: 18, height: 18,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 10, fontWeight: 700, padding: "0 3px", flexShrink: 0,
          }}>
            {alertCount > 99 ? "99+" : alertCount}
          </span>
        )}
      </button>
    ))}
  </aside>
);

// ═══════════════════════════════════════════════════════════════════════════════
// PANEL: Vessel Map
// ═══════════════════════════════════════════════════════════════════════════════
const VesselMapPanel = () => {
  const [filters, setFilters] = useState({ risk_level: "", vessel_type: "" });
  const { vessels, loading } = useVessels({ ...filters, limit: 3000 });
  const [selected, setSelected] = useState(null);
  const predPos = selected
    ? predictPosition(selected.lat, selected.lon, selected.sog, selected.heading || selected.cog || 0)
    : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 12 }}>
      {/* Filter bar */}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <select style={selectStyle}
          value={filters.risk_level}
          onChange={e => setFilters(f => ({ ...f, risk_level: e.target.value }))}>
          <option value="">All Risk Levels</option>
          <option value="HIGH">🔴 HIGH</option>
          <option value="MEDIUM">🟡 MEDIUM</option>
          <option value="LOW">🟢 LOW</option>
        </select>
        <select style={selectStyle}
          value={filters.vessel_type}
          onChange={e => setFilters(f => ({ ...f, vessel_type: e.target.value }))}>
          <option value="">All Types</option>
          {VESSEL_TYPE_OPTIONS.map(t => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>
        <span style={{ color: "#8B949E", fontSize: 13 }}>
          {loading ? "Loading..." : `${vessels.length.toLocaleString()} vessels`}
        </span>
      </div>

      {/* KPI row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
        <MetricCard label="Total Vessels" value={vessels.length} icon="🚢" />
        <MetricCard label="HIGH Risk"
          value={vessels.filter(v => v.risk_level==="HIGH").length}
          color="#EF5350" icon="🔴" />
        <MetricCard label="Anomalies"
          value={vessels.filter(v => v.is_anomaly).length}
          color="#FFA726" icon="⚠️" />
        <MetricCard label="Avg Speed"
          value={vessels.length
            ? `${(vessels.reduce((a,v)=>a+(v.sog||0),0)/vessels.length).toFixed(1)} kn`
            : "—"}
          icon="💨" />
      </div>

      {/* Map */}
      <div style={{ flex: 1, borderRadius: 12, overflow: "hidden", minHeight: 460 }}>
        <MapContainer center={[38.5, -75.5]} zoom={6}
          style={{ height: "100%", width: "100%" }}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {vessels
            .map(v => ({ ...v, lat: Number(v.lat), lon: Number(v.lon) }))
            .filter(v => Number.isFinite(v.lat) && Number.isFinite(v.lon))
            .map(v => (
            <Marker key={v.mmsi}
              position={[v.lat, v.lon]}
              icon={makeVesselIcon(
                v.heading || v.cog || 0,
                RISK_COLOR[v.risk_level] || "#90A4AE"
              )}
              eventHandlers={{ click: () => setSelected(v) }}>
              <Popup><VesselPopup v={v} /></Popup>
            </Marker>
          ))}
          {predPos && selected && (
            <>
              <Polyline
                positions={[[selected.lat, selected.lon], [predPos.lat, predPos.lon]]}
                pathOptions={{ color: "#42A5F5", weight: 2, dashArray: "6 4", opacity: 0.85 }}
              />
              <CircleMarker
                center={[predPos.lat, predPos.lon]}
                radius={8}
                pathOptions={{ color: "#42A5F5", fill: false, weight: 2 }}
              >
                <Popup>
                  <b>Predicted Position (5 min)</b><br />
                  {selected.vessel_name || selected.mmsi}<br />
                  {predPos.lat.toFixed(5)}°, {predPos.lon.toFixed(5)}°<br />
                  {haversineNm(selected.lat, selected.lon, predPos.lat, predPos.lon).toFixed(3)} nm ahead
                </Popup>
              </CircleMarker>
            </>
          )}
        </MapContainer>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PANEL: Historical Replay
// ═══════════════════════════════════════════════════════════════════════════════
const DEMO_MMSIS = [
  { mmsi: "366082000", name: "OVERSEAS SUN COAST", note: "276 records, ~2080 nm" },
  { mmsi: "367060370", name: "EVEY T",             note: "217 records, ~1200 nm" },
  { mmsi: "368530000", name: "C HERO",             note: "180 records, ~2335 nm" },
];

// Bearing (degrees, 0 = north) from point p1 to p2
const computeBearing = (p1, p2) => {
  const toRad = d => d * Math.PI / 180;
  const dLon = toRad(p2.lon - p1.lon);
  const lat1 = toRad(p1.lat), lat2 = toRad(p2.lat);
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return ((Math.atan2(y, x) * 180 / Math.PI) + 360) % 360;
};

// Inline SVG ship icon for the replay marker, rotated to face direction of travel
const makeReplayShipIcon = (bearing) => {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="38" height="38" viewBox="0 0 28 28">
      <g transform="rotate(${bearing}, 14, 14)">
        <path d="M14 2 C16 3.5 17 6 17 8.5 L17 23 Q17 26 14 26 Q11 26 11 23 L11 8.5 C11 6 12 3.5 14 2 Z" fill="#1A3A8F" stroke="white" stroke-width="0.9"/>
        <path d="M14 4.5 C15.4 5.6 16 7.4 16 9 L16 22 Q16 23.8 14 23.8 Q12 23.8 12 22 L12 9 C12 7.4 12.6 5.6 14 4.5 Z" fill="#4F6FD0"/>
        <rect x="12.4" y="15.5" width="3.2" height="4.2" rx="0.5" fill="white"/>
        <rect x="12.8" y="16.2" width="2.4" height="0.9" rx="0.2" fill="#263238"/>
        <rect x="12.6" y="9.5" width="2.8" height="1.3" rx="0.3" fill="#0D2A6B"/>
        <rect x="12.6" y="11.6" width="2.8" height="1.3" rx="0.3" fill="#0D2A6B"/>
      </g>
    </svg>`;
  return L.divIcon({ html: svg, className: "", iconSize: [38, 38], iconAnchor: [19, 19] });
};

// Smooth flyTo the track's first point at zoom 10 when the track changes
const MapFlyTo = ({ points }) => {
  const map = useMap();
  useEffect(() => {
    if (points && points.length > 0) {
      map.flyTo([points[0].lat, points[0].lon], 10);
    }
  }, [points, map]);
  return null;
};

const ReplayPanel = () => {
  const [mmsi,         setMmsi]         = useState("");
  const [track,        setTrack]        = useState([]);
  const [frame,        setFrame]        = useState(0);
  const [playing,      setPlaying]      = useState(false);
  const [speed,        setSpeed]        = useState(1);
  const [loadingTrack, setLoadingTrack] = useState(false);
  const [loaded,       setLoaded]       = useState(false);
  const timer = useRef(null);

  const loadTrack = async (overrideMmsi) => {
    const target = overrideMmsi || mmsi;
    if (!target) return;
    setLoadingTrack(true);
    setLoaded(false);
    setPlaying(false);
    const d = await apiFetch(`/api/vessels/${target}/track`, { limit: 5000 });
    if (d && d.points && d.points.length > 0) {
      setTrack(d.points);
      setFrame(0);
    } else {
      setTrack([]);
    }
    setLoaded(true);
    setLoadingTrack(false);
  };

  useEffect(() => {
    if (playing && track.length > 0) {
      timer.current = setInterval(() => {
        setFrame(f => {
          if (f >= track.length - 1) { setPlaying(false); return f; }
          return f + 1;
        });
      }, 500 / speed);
    } else {
      clearInterval(timer.current);
    }
    return () => clearInterval(timer.current);
  }, [playing, speed, track]);

  const curr = track[frame];
  const replayBearing = curr && track[frame + 1]
    ? computeBearing(curr, track[frame + 1])
    : (curr?.cog || curr?.heading || 0);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 12 }}>
      {/* Controls */}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <input style={{ ...inputStyle, width: 160 }}
          placeholder="Enter MMSI e.g. 366082000"
          value={mmsi}
          onChange={e => setMmsi(e.target.value)}
          onKeyDown={e => e.key === "Enter" && loadTrack()} />
        <button
          style={{ ...btnStyle, opacity: loadingTrack ? 0.6 : 1 }}
          onClick={() => loadTrack()}
          disabled={loadingTrack}>
          {loadingTrack ? "Loading…" : "Load Track"}
        </button>
        <button
          style={{ ...btnStyle, background: playing ? "#b45309" : "#166534" }}
          onClick={() => setPlaying(p => !p)}
          disabled={track.length === 0}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <span style={{ color: "#8B949E", fontSize: 13 }}>Speed</span>
        <input type="range" min={1} max={10} value={speed}
          onChange={e => setSpeed(+e.target.value)}
          style={{ width: 80 }} />
        <span style={{ color: "#fff", fontSize: 13 }}>{speed}×</span>
        {track.length > 0 &&
          <span style={{ color: "#8B949E", fontSize: 13 }}>
            Frame {frame + 1}/{track.length}
          </span>}
      </div>

      {/* Empty-state with MMSI hints */}
      {loaded && track.length === 0 && (
        <div style={{
          background: "#161B22", border: "1px solid #30363D",
          borderRadius: 8, padding: "14px 16px",
        }}>
          <p style={{ color: "#FFA726", margin: "0 0 8px", fontWeight: 600 }}>
            No track data for MMSI {mmsi}.
          </p>
          <p style={{ color: "#8B949E", margin: "0 0 10px", fontSize: 13 }}>
            Try one of these vessels with confirmed movement:
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {DEMO_MMSIS.map(d => (
              <button key={d.mmsi}
                style={{ ...btnStyle, background: "#21262D", fontSize: 12 }}
                onClick={() => { setMmsi(d.mmsi); loadTrack(d.mmsi); }}>
                {d.mmsi} — {d.name} ({d.note})
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Timeline slider */}
      {track.length > 0 && (
        <input type="range" min={0} max={track.length - 1} value={frame}
          onChange={e => setFrame(+e.target.value)}
          style={{ width: "100%", accentColor: "#3B82F6" }} />
      )}

      {/* Current frame info */}
      {curr && (
        <div style={{ color: "#8B949E", fontSize: 13 }}>
          🕒 {curr.base_datetime?.slice(0, 19)} |{" "}
          📍 ({curr.lat?.toFixed(4)}, {curr.lon?.toFixed(4)}) |{" "}
          💨 {curr.sog?.toFixed(1)} kn |{" "}
          Risk:{" "}
          <span style={{ color: RISK_COLOR[curr.risk_level] || "#8B949E" }}>
            {curr.risk_level || "—"}
          </span>
        </div>
      )}

      {/* Map */}
      <div style={{ flex: 1, borderRadius: 12, overflow: "hidden", minHeight: 420 }}>
        <MapContainer
          center={track.length > 0 ? [track[0].lat, track[0].lon] : [38.5, -95.0]}
          zoom={track.length > 0 ? 10 : 4}
          style={{ height: "100%", width: "100%" }}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {/* Fly to first track point (zoom 10) whenever a new track loads */}
          {track.length > 0 && <MapFlyTo points={track} />}
          {/* Full route polyline (ghost track) */}
          {track.length > 0 && (
            <Polyline positions={track.map(p => [p.lat, p.lon])}
              color="#42A5F5" weight={2} opacity={0.25} />
          )}
          {/* Growing trail — traveled portion only */}
          {frame > 0 && (
            <Polyline
              positions={track.slice(0, frame + 1).map(p => [p.lat, p.lon])}
              color="#FF5252" weight={3} opacity={0.85}
            />
          )}
          {/* Animated ship icon, rotated to face direction of travel */}
          {curr && (
            <Marker
              position={[curr.lat, curr.lon]}
              icon={makeReplayShipIcon(replayBearing)}>
              <Popup>
                SOG: {curr.sog?.toFixed(1)} kn<br />
                {curr.base_datetime?.slice(0, 19)}
              </Popup>
            </Marker>
          )}
        </MapContainer>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PANEL: Analytics
// ═══════════════════════════════════════════════════════════════════════════════
const AnalyticsPanel = () => {
  const data = useAnalytics();
  if (!data) return <div style={centerStyle}>Loading analytics...</div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20,
                  overflow: "auto", paddingRight: 4 }}>
      {/* KPI */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
        <MetricCard label="Total Vessels"  value={data.total_vessels?.toLocaleString()}  icon="🚢" />
        <MetricCard label="Active Alerts"  value={data.total_alerts}   icon="🔔" color="#FFA726" />
        <MetricCard label="HIGH Risk"      value={data.high_risk_count} icon="🔴" color="#EF5350" />
        <MetricCard label="Avg Speed"      value={`${data.avg_speed_kn} kn`} icon="💨" color="#42A5F5" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Vessel types */}
        <div style={cardStyle}>
          <h3 style={cardTitle}>Vessel Types</h3>
          <ResponsiveContainer width="100%" height={230}>
            <PieChart>
              <Pie data={data.vessel_types || []}
                dataKey="count" nameKey="type"
                cx="50%" cy="50%" outerRadius={80}
                label={({ type }) => type}>
                {(data.vessel_types || []).map((_, i) => (
                  <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: "#1F2937", border: "none" }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Alert types */}
        <div style={cardStyle}>
          <h3 style={cardTitle}>Alert Types (14 days)</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={data.alert_types || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="type" tick={{ fill: "#9CA3AF", fontSize: 10 }} />
              <YAxis tick={{ fill: "#9CA3AF", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#1F2937", border: "none" }} />
              <Bar dataKey="count" fill="#3B82F6" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Daily vessel count */}
        {data.daily_stats && data.daily_stats.length > 0 && (
          <div style={{ ...cardStyle, gridColumn: "1 / -1" }}>
            <h3 style={cardTitle}>Daily Vessel Count</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={data.daily_stats}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" tick={{ fill: "#9CA3AF", fontSize: 11 }} />
                <YAxis tick={{ fill: "#9CA3AF", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#1F2937", border: "none" }} />
                <Legend />
                <Line type="monotone" dataKey="total_vessels"
                  stroke="#42A5F5" name="Vessels" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="high_risk"
                  stroke="#EF5350" name="HIGH Risk" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PANEL: Alerts  (receives alerts + lastUpdate from App-level useAlerts)
// ═══════════════════════════════════════════════════════════════════════════════
const AlertsPanel = ({ alerts, lastUpdate }) => {
  const resolve = async (id) => {
    await fetch(`${API}/api/alerts/${id}/resolve`, { method: "PATCH" });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <h2 style={{ color: "#fff", fontSize: 20, fontWeight: 700, margin: 0 }}>
          🔔 Alerts ({alerts.length})
        </h2>
        {/* Live indicator */}
        <span style={{
          display: "flex", alignItems: "center", gap: 5,
          color: "#3FB950", fontSize: 12, fontWeight: 600,
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: "50%", background: "#3FB950",
            display: "inline-block", animation: "pulse 2s infinite",
          }} />
          LIVE — updating every 3s
        </span>
        {lastUpdate && (
          <span style={{ color: "#6e7681", fontSize: 11 }}>
            Last: {lastUpdate.toLocaleTimeString()}
          </span>
        )}
        <span style={{ color: "#6e7681", fontSize: 11 }}>
          Showing last 1 hour
        </span>
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {["CRITICAL","HIGH","MEDIUM","LOW"].map(s => (
          <MetricCard key={s} label={s}
            value={alerts.filter(a => a.severity === s).length}
            color={RISK_COLOR[s] || RISK_COLOR.LOW}
            icon={s==="CRITICAL"||s==="HIGH"?"🔴":s==="MEDIUM"?"🟡":"🟢"} />
        ))}
      </div>

      <div style={{ overflow: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ background: "#161B22" }}>
              {["Type","Severity","Vessel","Description","Time","Action"].map(h => (
                <th key={h} style={thStyle}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {alerts.map(a => (
              <tr key={a.id}
                style={{
                  borderBottom: "1px solid #21262D",
                  background: a.severity === "CRITICAL" ? "#1a0808"
                            : a.severity === "HIGH"     ? "#1a1008" : "transparent",
                }}>
                <td style={tdStyle}>
                  <span style={{ fontFamily: "monospace", fontSize: 11 }}>{a.type}</span>
                </td>
                <td style={tdStyle}><Badge severity={a.severity} /></td>
                <td style={tdStyle}>{a.vessel_name || a.mmsi || "—"}</td>
                <td style={{ ...tdStyle, maxWidth: 300, overflow: "hidden",
                             textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {a.description}
                </td>
                <td style={{ ...tdStyle, color: "#6e7681", fontSize: 11 }}>
                  {a.created_at?.slice(11, 19)}
                </td>
                <td style={tdStyle}>
                  {!a.is_resolved && (
                    <button onClick={() => resolve(a.id)}
                      style={{ ...btnStyle, padding: "3px 8px", fontSize: 11 }}>
                      ✓
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {alerts.length === 0 &&
          <div style={centerStyle}>No alerts in the last hour ✅</div>}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PANEL: Search
// ═══════════════════════════════════════════════════════════════════════════════
const SearchPanel = () => {
  const [q,       setQ]       = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const search = async () => {
    if (q.length < 2) return;
    setLoading(true);
    const d = await apiFetch("/api/search", { q, limit: 100 });
    if (d) setResults(d.results || []);
    setLoading(false);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <h2 style={{ color: "#fff", fontSize: 20, fontWeight: 700, margin: 0 }}>
        🔍 Vessel Search
      </h2>
      <div style={{ display: "flex", gap: 10 }}>
        <input style={{ ...inputStyle, flex: 1 }}
          placeholder="Search MMSI, vessel name, IMO, call sign..."
          value={q} onChange={e => setQ(e.target.value)}
          onKeyDown={e => e.key === "Enter" && search()} />
        <button style={btnStyle} onClick={search}>
          {loading ? "..." : "Search"}
        </button>
      </div>

      {results.length > 0 && (
        <>
          <p style={{ color: "#8B949E", fontSize: 13 }}>
            Found {results.length} vessel(s)
          </p>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ background: "#161B22" }}>
                {["MMSI","Name","Type","Speed","Risk","Anomaly"].map(h => (
                  <th key={h} style={thStyle}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map(v => (
                <tr key={v.mmsi} style={{ borderBottom: "1px solid #21262D" }}>
                  <td style={{ ...tdStyle, fontFamily: "monospace", fontSize: 11 }}>
                    {v.mmsi}
                  </td>
                  <td style={tdStyle}>{v.vessel_name || "—"}</td>
                  <td style={tdStyle}>{v.vessel_type_label || "—"}</td>
                  <td style={tdStyle}>{(v.sog||0).toFixed(1)} kn</td>
                  <td style={tdStyle}>
                    <span style={{ color: RISK_COLOR[v.risk_level] }}>
                      {v.risk_level}
                    </span>
                  </td>
                  <td style={tdStyle}>
                    {v.is_anomaly
                      ? <span style={{ color: "#EF5350" }}>⚠️ Yes</span>
                      : <span style={{ color: "#66BB6A" }}>✓ No</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PANEL: Traffic Heatmap
// ═══════════════════════════════════════════════════════════════════════════════
const TIME_OPTS = [
  { label: "Last 1h",  value: 1  },
  { label: "Last 6h",  value: 6  },
  { label: "Last 24h", value: 24 },
  { label: "All Historical", value: 10000 }
];

const HeatmapPanel = () => {
  const [cells,      setCells]      = useState([]);
  const [loading,    setLoading]    = useState(true);
  const [hoursBack,  setHoursBack]  = useState(10000);
  const [isFallback, setIsFallback] = useState(false);

  useEffect(() => {
    let active = true;
    const load = async () => {
      setLoading(true);
      const d = await apiFetch("/api/density", { hours_back: hoursBack, min_vessels: 1 });
      if (d && active) {
        setCells(d.cells || []);
        setIsFallback(d.is_historical_fallback || false);
      }
      if (active) setLoading(false);
    };
    load();
    const t = setInterval(load, 30000);
    return () => { active = false; clearInterval(t); };
  }, [hoursBack]);

  const highZones = cells.filter(c => c.congestion_level === "HIGH").length;
  const medZones  = cells.filter(c => c.congestion_level === "MEDIUM").length;
  // cells are sorted by vessel_count desc from the API
  const topCell   = cells.length > 0 ? cells[0] : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 12 }}>

      {/* Time filter */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span style={{ color: "#8B949E", fontSize: 13, fontWeight: 600 }}>
          Time window:
        </span>
        {TIME_OPTS.map(opt => (
          <button key={opt.value}
            onClick={() => setHoursBack(opt.value)}
            style={{
              ...btnStyle,
              background: hoursBack === opt.value ? "#1F6FEB" : "#21262D",
              border: `1px solid ${hoursBack === opt.value ? "#1F6FEB" : "#30363D"}`,
              padding: "5px 14px", fontSize: 12,
            }}>
            {opt.label}
          </button>
        ))}
        {isFallback && (
          <span style={{
            color: "#FFA726", fontSize: 12,
            background: "#78350f22", border: "1px solid #78350f",
            borderRadius: 6, padding: "3px 10px",
          }}>
            Showing historical AIS data for demo
          </span>
        )}
        {loading && (
          <span style={{ color: "#6e7681", fontSize: 12 }}>Loading…</span>
        )}
      </div>

      {/* Metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
        <MetricCard
          label="HIGH Congestion"
          value={highZones}
          icon="🔴" color="#EF5350" />
        <MetricCard
          label="MEDIUM Congestion"
          value={medZones}
          icon="🟡" color="#FFA726" />
        <MetricCard
          label="Grid Cells"
          value={cells.length}
          icon="📐" />
        <MetricCard
          label="Most Congested"
          value={topCell ? `${topCell.vessel_count.toLocaleString()} vessels` : "—"}
          sub={topCell ? `${topCell.lat?.toFixed(1)}°, ${topCell.lon?.toFixed(1)}°` : undefined}
          icon="🏙️" color="#AB47BC" />
      </div>

      {/* Map */}
      <div style={{ flex: 1, borderRadius: 12, overflow: "hidden", minHeight: 460 }}>
        <MapContainer center={[47.5, -122.3]} zoom={6}
          style={{ height: "100%", width: "100%" }}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {cells.map((c, i) => (
            <CircleMarker key={i}
              center={[c.lat, c.lon]}
              radius={Math.max(4, Math.min(30, Math.sqrt(c.vessel_count || 1) * 1.5))}
              pathOptions={{
                color: c.congestion_level === "HIGH"   ? "#EF5350"
                     : c.congestion_level === "MEDIUM" ? "#FFA726" : "#42A5F5",
                fillColor: c.congestion_level === "HIGH"   ? "#EF5350"
                         : c.congestion_level === "MEDIUM" ? "#FFA726" : "#42A5F5",
                fillOpacity: Math.min(0.85, 0.3 + (c.weight || 0) * 0.55),
                weight: 0,
              }}>
              <Popup>
                <b>Density Cell</b><br />
                Vessels: {(c.vessel_count || 0).toLocaleString()}<br />
                Avg Speed: {(c.avg_sog || 0).toFixed(1)} kn<br />
                Level: <b>{c.congestion_level}</b>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>

      {!loading && cells.length === 0 && (
        <div style={centerStyle}>No density data. Ensure gold_job has run.</div>
      )}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PANEL: Anomaly Detection  (polls every 3s, live indicator, flashes on new)
// ═══════════════════════════════════════════════════════════════════════════════
const AnomalyPanel = () => {
  const [anomalies, setAnomalies] = useState([]);
  const [loading,   setLoading]   = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [flash, setFlash] = useState(false);
  const prevCount = useRef(0);

  useEffect(() => {
    const load = async () => {
      const d = await apiFetch("/api/anomalies", { hours_back: 24, limit: 200 });
      if (d) {
        const next = d.anomalies || [];
        // Flash red border when new anomalies arrive (skip first load)
        if (prevCount.current > 0 && next.length > prevCount.current) {
          setFlash(true);
          setTimeout(() => setFlash(false), 800);
        }
        prevCount.current = next.length;
        setAnomalies(next);
        setLastUpdate(new Date());
      }
      setLoading(false);
    };
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []);

  const ANOM_COLOR = {
    ML_ANOMALY: "#EF5350", SUDDEN_STOP: "#FF7043",
    SHARP_TURN: "#FFA726", UNUSUAL_SPEED: "#AB47BC",
    STATIONARY_RISK: "#FF5722", UNEXPECTED_DIRECTION: "#F44336",
  };

  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 12,
      border: flash ? "2px solid #EF5350" : "2px solid transparent",
      borderRadius: 10, padding: 2,
      transition: "border-color 0.15s ease",
    }}>
      {/* Live indicator row */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span style={{
          display: "flex", alignItems: "center", gap: 6,
          color: "#3FB950", fontSize: 12, fontWeight: 600,
        }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%", background: "#3FB950",
            display: "inline-block", animation: "pulse 1.5s infinite",
          }} />
          LIVE — updating every 3s
        </span>
        {lastUpdate && (
          <span style={{ color: "#6e7681", fontSize: 11 }}>
            Last update: {lastUpdate.toLocaleTimeString()}
          </span>
        )}
        {flash && (
          <span style={{ color: "#EF5350", fontSize: 12, fontWeight: 700,
                         animation: "pulse 0.5s ease" }}>
            ● New anomaly detected
          </span>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
        <MetricCard label="Total Anomalies" value={anomalies.length}                                      icon="⚠️" color="#FFA726" />
        <MetricCard label="HIGH Severity"   value={anomalies.filter(a => a.severity==="HIGH").length}     icon="🔴" color="#EF5350" />
        <MetricCard label="MEDIUM"          value={anomalies.filter(a => a.severity==="MEDIUM").length}   icon="🟡" color="#FFA726" />
        <MetricCard label="ML Detected"     value={anomalies.filter(a => a.type==="ML_ANOMALY").length}   icon="🤖" color="#AB47BC" />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, flex: 1 }}>
        <div style={{ borderRadius: 12, overflow: "hidden", minHeight: 400 }}>
          <MapContainer center={[38.5, -75.5]} zoom={6}
            style={{ height: "100%", width: "100%" }}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            {anomalies.filter(a => a.lat != null && a.lon != null).map(a => (
              <CircleMarker key={a.id} center={[a.lat, a.lon]}
                radius={a.severity === "HIGH" ? 10 : a.severity === "MEDIUM" ? 8 : 6}
                pathOptions={{
                  color: a.severity === "HIGH" ? "#EF5350" : a.severity === "MEDIUM" ? "#FFA726" : "#4CAF50",
                  fillColor: a.severity === "HIGH" ? "#EF5350" : a.severity === "MEDIUM" ? "#FFA726" : "#4CAF50",
                  fillOpacity: 0.85, weight: 1,
                }}>
                <Popup>
                  <b>{a.type}</b><br />
                  Severity: {a.severity}<br />
                  MMSI: {a.mmsi}<br />
                  Score: {(a.score || 0).toFixed(2)}<br />
                  {a.description}
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
        </div>
        <div style={{ overflow: "auto", maxHeight: 420 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ background: "#161B22" }}>
                {["Type","Severity","MMSI","Score","Time"].map(h => (
                  <th key={h} style={thStyle}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {anomalies.map(a => (
                <tr key={a.id} style={{ borderBottom: "1px solid #21262D" }}>
                  <td style={{ ...tdStyle, fontSize: 11, fontFamily: "monospace" }}>{a.type}</td>
                  <td style={tdStyle}><Badge severity={a.severity} /></td>
                  <td style={{ ...tdStyle, fontFamily: "monospace", fontSize: 11 }}>{a.mmsi}</td>
                  <td style={{ ...tdStyle, color: "#FFA726" }}>{(a.score || 0).toFixed(2)}</td>
                  <td style={{ ...tdStyle, color: "#6e7681", fontSize: 11 }}>{a.created_at?.slice(11,19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!loading && anomalies.length === 0 && (
            <div style={centerStyle}>No anomalies in last 24h.<br />Start the live scorer and producer.</div>
          )}
        </div>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PANEL: Collision Risk  (polls every 3s, calls onNewCritical for banner)
// ═══════════════════════════════════════════════════════════════════════════════
const CollisionPanel = ({ onNewCritical }) => {
  const [risks,   setRisks]   = useState([]);
  const [loading, setLoading] = useState(true);
  // stable ref so the effect doesn't need to re-run when callback identity changes
  const onNewCriticalRef = useRef(onNewCritical);
  useEffect(() => { onNewCriticalRef.current = onNewCritical; }, [onNewCritical]);
  const seenCriticalIds = useRef(new Set());

  useEffect(() => {
    const load = async () => {
      const d = await apiFetch("/api/collision-risks", { hours_back: 6 });
      const newRisks = d?.risks || [];
      setRisks(newRisks);
      setLoading(false);

      // Fire banner for any CRITICAL not seen in previous polls
      const newCriticals = newRisks.filter(
        r => r.severity === "CRITICAL" && !seenCriticalIds.current.has(r.id)
      );
      if (newCriticals.length > 0) {
        onNewCriticalRef.current?.(newCriticals[0]);
      }
      newRisks.forEach(r => seenCriticalIds.current.add(r.id));
    };
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []); // empty — uses refs for stable access

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
        <MetricCard label="Active Risks" value={risks.length}                                        icon="🚨" color="#EF5350" />
        <MetricCard label="CRITICAL"     value={risks.filter(r=>r.severity==="CRITICAL").length}     icon="💀" color="#B71C1C" />
        <MetricCard label="HIGH"         value={risks.filter(r=>r.severity==="HIGH").length}         icon="🔴" color="#EF5350" />
        <MetricCard label="MEDIUM"       value={risks.filter(r=>r.severity==="MEDIUM").length}       icon="🟡" color="#FFA726" />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div style={{ borderRadius: 12, overflow: "hidden", minHeight: 380 }}>
          <MapContainer center={[38.5, -75.5]} zoom={6}
            style={{ height: "100%", width: "100%" }}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            {risks.filter(r => r.lat1 != null && r.lon1 != null && r.lat2 != null && r.lon2 != null).map(r => (
              <React.Fragment key={r.id}>
                <Polyline
                  positions={[[r.lat1, r.lon1], [r.lat2, r.lon2]]}
                  pathOptions={{ color: RISK_COLOR[r.severity] || "#EF5350", weight: 2, dashArray: "5,5" }}
                />
                <CircleMarker center={[r.lat1, r.lon1]}
                  radius={r.severity === "CRITICAL" ? 12 : r.severity === "HIGH" ? 9 : 6}
                  pathOptions={{ color: RISK_COLOR[r.severity] || "#EF5350", fillColor: RISK_COLOR[r.severity] || "#EF5350", fillOpacity: 0.9, weight: 2 }}>
                  <Popup>
                    <b>🚨 {r.severity}</b><br />
                    {r.vessel_1 || r.mmsi_1}<br />
                    Distance: {(r.distance_nm || 0).toFixed(3)} nm<br />
                    {r.description}
                  </Popup>
                </CircleMarker>
                <CircleMarker center={[r.lat2, r.lon2]}
                  radius={r.severity === "CRITICAL" ? 12 : r.severity === "HIGH" ? 9 : 6}
                  pathOptions={{ color: RISK_COLOR[r.severity] || "#EF5350", fillColor: RISK_COLOR[r.severity] || "#EF5350", fillOpacity: 0.9, weight: 2 }}>
                  <Popup>
                    <b>🚨 {r.severity}</b><br />
                    {r.vessel_2 || r.mmsi_2}<br />
                    Distance: {(r.distance_nm || 0).toFixed(3)} nm<br />
                    {r.description}
                  </Popup>
                </CircleMarker>
              </React.Fragment>
            ))}
            {risks.filter(r => (r.lat1 == null || r.lon1 == null) && r.lat != null && r.lon != null).map(r => (
              <CircleMarker key={r.id} center={[r.lat, r.lon]}
                radius={r.severity === "CRITICAL" ? 12 : r.severity === "HIGH" ? 9 : 6}
                pathOptions={{ color: RISK_COLOR[r.severity] || "#EF5350", fillColor: RISK_COLOR[r.severity] || "#EF5350", fillOpacity: 0.9, weight: 2 }}>
                <Popup>
                  <b>🚨 {r.severity}</b><br />
                  {r.vessel_1 || r.mmsi_1} ↔ {r.vessel_2 || r.mmsi_2}<br />
                  Distance: {(r.distance_nm || 0).toFixed(3)} nm<br />
                  {r.description}
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
        </div>
        <div style={{ overflow: "auto", maxHeight: 380 }}>
          {!loading && risks.length === 0 ? (
            <div style={centerStyle}>✅ No collision risks in last 6h.</div>
          ) : (
            risks.map(r => (
              <div key={r.id} style={{
                background: r.severity === "CRITICAL" ? "#1a0808" : "#1a1008",
                border: `1px solid ${RISK_COLOR[r.severity] || "#374151"}`,
                borderRadius: 8, padding: "10px 14px", marginBottom: 8,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <Badge severity={r.severity} />
                  <span style={{ color: "#6e7681", fontSize: 11 }}>
                    {(r.distance_nm || 0).toFixed(3)} nm apart
                  </span>
                </div>
                <div style={{ color: "#E6EDF3", fontSize: 13 }}>
                  <b>{r.vessel_1 || r.mmsi_1}</b> ↔ <b>{r.vessel_2 || r.mmsi_2}</b>
                </div>
                <div style={{ color: "#8B949E", fontSize: 12, marginTop: 4 }}>
                  {r.description}
                </div>
                <div style={{ color: "#6e7681", fontSize: 11, marginTop: 4 }}>
                  {r.created_at?.slice(0, 19)}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

// ── Shared styles ──────────────────────────────────────────────────────────────
const selectStyle = {
  background: "#21262D", color: "#E6EDF3",
  border: "1px solid #30363D", borderRadius: 6,
  padding: "6px 10px", fontSize: 13,
};
const inputStyle = {
  background: "#21262D", color: "#E6EDF3",
  border: "1px solid #30363D", borderRadius: 6,
  padding: "8px 12px", fontSize: 13, outline: "none",
};
const btnStyle = {
  background: "#1F6FEB", color: "#fff",
  border: "none", borderRadius: 6,
  padding: "8px 16px", fontSize: 13,
  cursor: "pointer", fontWeight: 600,
};
const cardStyle = {
  background: "#161B22", borderRadius: 10,
  padding: "16px", border: "1px solid #21262D",
};
const cardTitle = {
  color: "#E6EDF3", fontSize: 15, fontWeight: 600,
  margin: "0 0 12px",
};
const centerStyle = {
  textAlign: "center", color: "#8B949E",
  padding: "60px 0",
};
const thStyle = {
  padding: "8px 12px", color: "#8B949E",
  fontWeight: 600, textAlign: "left",
  fontSize: 12, borderBottom: "1px solid #21262D",
};
const tdStyle = {
  padding: "8px 12px", color: "#E6EDF3",
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [page, setPage] = useState("map");
  const [time, setTime] = useState(new Date());

  // Alerts hoisted here so both Sidebar badge and AlertsPanel share one poll
  const { alerts, lastUpdate: alertsLastUpdate } = useAlerts();
  const unresolvedCount = alerts.filter(a => !a.is_resolved).length;

  // CRITICAL collision banner
  const [criticalBanner, setCriticalBanner] = useState(null);
  const bannerTimer = useRef(null);
  const handleNewCritical = useCallback((risk) => {
    setCriticalBanner(risk);
    clearTimeout(bannerTimer.current);
    bannerTimer.current = setTimeout(() => setCriticalBanner(null), 10000);
  }, []);

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const panels = {
    map:      <VesselMapPanel />,
    replay:   <ReplayPanel />,
    analytics:<AnalyticsPanel />,
    alerts:   <AlertsPanel alerts={alerts} lastUpdate={alertsLastUpdate} />,
    search:   <SearchPanel />,
    heatmap:  <HeatmapPanel />,
    anomaly:  <AnomalyPanel />,
    collision:<CollisionPanel onNewCritical={handleNewCritical} />,
  };

  return (
    <div style={{
      display: "flex", height: "100vh",
      background: "#0D1117", color: "#E6EDF3",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      overflow: "hidden",
    }}>
      {/* Global CRITICAL collision banner — fixed top, auto-dismisses in 10s */}
      <CriticalBanner
        alert={criticalBanner}
        onDismiss={() => { setCriticalBanner(null); clearTimeout(bannerTimer.current); }}
      />

      <Sidebar active={page} onChange={setPage} alertCount={unresolvedCount} />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Header */}
        <header style={{
          background: "#161B22", borderBottom: "1px solid #21262D",
          padding: "12px 24px", display: "flex",
          alignItems: "center", justifyContent: "space-between",
        }}>
          <h1 style={{ fontSize: 17, fontWeight: 700, margin: 0 }}>
            🚢 Maritime Navigation AI System
          </h1>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <span style={{ color: "#3FB950", fontSize: 12, display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{
                width: 7, height: 7, borderRadius: "50%",
                background: "#3FB950", display: "inline-block",
                animation: "pulse 2s infinite",
              }} />
              Live
            </span>
            <span style={{ color: "#6e7681", fontSize: 12 }}>
              {time.toUTCString().slice(0, 25)} UTC
            </span>
          </div>
        </header>

        {/* Content */}
        <main style={{ flex: 1, padding: 20, overflow: "auto" }}>
          {panels[page]}
        </main>
      </div>
    </div>
  );
}
