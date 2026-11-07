import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronDown,
  Clock3,
  Download,
  FileText,
  Film,
  FolderOpen,
  Layers,
  Lock,
  LogOut,
  Play,
  Plus,
  ScanLine,
  Search,
  Settings2,
  SlidersHorizontal,
  VideoOff,
  Trash2,
  Upload,
  Users,
  Video,
  X,
} from "lucide-react";
import "./style.css";

type Point = [number, number];
type Zone = { name: string; capacity: number; polygon: Point[] };
type Settings = {
  location: string;
  zones: Zone[];
  conf: number;
  analysis_fps: number;
  event_seconds: number;
  imgsz: number;
};
type Event = {
  zone: string;
  start_seconds: number;
  end_seconds: number;
  duration_seconds: number;
  peak_seconds: number;
  peak_count: number;
  capacity: number;
};
type Result = {
  max_head_count: number;
  avg_head_count: number;
  processed_frames: number;
  timestamps: number[];
  per_frame_counts: number[];
  events: Event[];
  zones: (Zone & { counts: number[]; peak: number })[];
  heatmap: { grid: number[][]; samples: number; measurement: string };
  settings: Settings;
  input_bytes: number;
  output_bytes: number;
  processing_time_seconds: number;
};
type Session = {
  id: string;
  file_name: string;
  created: number;
  bytes: number;
  duration: number;
  status: {
    state: string;
    progress?: number;
    message?: string;
    error?: string;
  };
  result?: Result;
  settings?: Settings;
  media_type?: string;
};

type SystemStatus = {
  detector_ready: boolean;
  model_name: string;
  weights_file: string | null;
  deployment_mode: "local" | "hosted";
  max_upload_bytes: number;
  free_storage_bytes: number;
  total_storage_bytes: number;
  retention_days: number;
  active_jobs: number;
};

const defaults: Settings = {
  location: "Campus event",
  zones: [
    {
      name: "Main area",
      capacity: 80,
      polygon: [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ],
    },
  ],
  conf: 0.35,
  analysis_fps: 5,
  event_seconds: 3,
  imgsz: 640,
};

type AppPage = "Review Footage" | "Live Monitoring" | "Activity & History" | "Reports";

const active = (s: Session) =>
  ["queued", "running", "cancelling"].includes(s.status.state);
const fmt = (t: number) =>
  `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const mb = (b: number) => (b / 1024 ** 2).toFixed(1) + " MB";

async function api(path: string, method = "GET", body?: unknown) {
  const r = await fetch("/api" + path, {
    method,
    headers: { "Content-Type": "application/json", "X-Campus-Request": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    const e = await r
      .json()
      .catch(() => ({ detail: "Server unavailable. Try again." }));
    throw Error(
      typeof e.detail === "string" ? e.detail : "Check the entered settings.",
    );
  }
  return r.json();
}

function normalizePage(stored: string | null): AppPage {
  if (stored === "Live Monitoring") return "Live Monitoring";
  if (stored === "Activity & History" || stored === "Sessions") return "Activity & History";
  if (stored === "Reports") return "Reports";
  return "Review Footage";
}

function App() {
  const [page, setPage] = useState<AppPage>(() =>
    normalizePage(sessionStorage.getItem("campus-page")),
  );
  const [auth, setAuth] = useState<boolean | null>(null);
  const [password, setPassword] = useState("");
  const [retention, setRetention] = useState(7);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<Session | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [footageSettingsOpen, setFootageSettingsOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);

  // Live monitoring local settings
  const [liveSourceType, setLiveSourceType] = useState("rtsp");
  const [rtspUrl, setRtspUrl] = useState(() => localStorage.getItem("campus-rtsp-url") || "");
  const [liveLocation, setLiveLocation] = useState("Campus entrance");
  const [liveCapacity, setLiveCapacity] = useState(80);
  const [cameraCheck, setCameraCheck] = useState<{
    tested: boolean;
    connected: boolean;
    checking: boolean;
    message: string;
    width?: number;
    height?: number;
    fps?: number;
  }>({
    tested: false,
    connected: false,
    checking: false,
    message: "Camera not connected. Enter an RTSP address or camera index and click 'Test Connection'.",
  });

  const [draft, setDraft] = useState<Settings>(() => {
    try {
      return (
        JSON.parse(localStorage.getItem("campus-settings") || "null") ||
        defaults
      );
    } catch {
      return defaults;
    }
  });

  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const selectedId = useRef(sessionStorage.getItem("campus-session") || "");

  const navigate = (p: AppPage) => {
    setPage(p);
    sessionStorage.setItem("campus-page", p);
    window.scrollTo(0, 0);
  };

  const refresh = async () => {
    try {
      const [s, sys] = await Promise.all([
        api("/sessions"),
        api("/system/status").catch(() => null),
      ]);
      setSessions(s);
      if (sys) setSystemStatus(sys);
    } catch {
      const s = await api("/sessions");
      setSessions(s);
    }
    if (selectedId.current) {
      try {
        setSelected(await api("/sessions/" + selectedId.current));
      } catch {
        selectedId.current = "";
        setSelected(null);
      }
    }
  };

  const testCameraConnection = async () => {
    const target = rtspUrl.trim();
    if (!target) {
      setCameraCheck({
        tested: true,
        connected: false,
        checking: false,
        message: "Please enter an RTSP stream address or camera device index before testing.",
      });
      return;
    }
    setCameraCheck((prev) => ({
      ...prev,
      checking: true,
      message: "Probing camera stream via backend OpenCV engine...",
    }));
    try {
      const res = await api("/camera/check", "POST", { source: target });
      setCameraCheck({
        tested: true,
        connected: !!res.connected,
        checking: false,
        message: res.message,
        width: res.width,
        height: res.height,
        fps: res.fps,
      });
    } catch (e) {
      setCameraCheck({
        tested: true,
        connected: false,
        checking: false,
        message: (e as Error).message || "Camera connection check failed.",
      });
    }
  };

  useEffect(() => {
    api("/auth")
      .then((x) => {
        setAuth(x.authenticated);
        setRetention(x.retention_days);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!auth) return;
    refresh().catch((e) => setError(e.message));
    const id = setInterval(
      () => refresh().catch((e) => setError(e.message)),
      2000,
    );
    return () => clearInterval(id);
  }, [auth]);

  useEffect(() => {
    localStorage.setItem("campus-settings", JSON.stringify(draft));
  }, [draft]);

  const action = async (fn: () => Promise<void>) => {
    setError("");
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const open = async (s: Session) => {
    selectedId.current = s.id;
    sessionStorage.setItem("campus-session", s.id);
    setSelected(await api("/sessions/" + s.id));
    navigate("Review Footage");
  };

  // Upload validated against real backend limits and disk constraints
  const upload = (file: File) =>
    action(async () => {
      const allowedExts = [".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png"];
      const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
      if (!allowedExts.includes(ext)) {
        throw Error("Unsupported format. Please upload an MP4, MOV, AVI, MKV recording or JPG, PNG image.");
      }
      const maxLimit = systemStatus?.max_upload_bytes || 50 * 1024 ** 3;
      if (file.size > maxLimit) {
        throw Error(
          `File size (${(file.size / 1024 ** 3).toFixed(2)} GB) exceeds the server upload limit of ${(maxLimit / 1024 ** 3).toFixed(0)} GB.`,
        );
      }
      if (systemStatus?.free_storage_bytes && file.size > systemStatus.free_storage_bytes) {
        throw Error("Insufficient server storage space to receive this file.");
      }
      setUploadProgress(0);
      try {
        const s = await new Promise<Session>((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          xhr.open("POST", "/api/upload");
          xhr.setRequestHeader("X-Campus-Request", "1");
          xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
          xhr.upload.onprogress = (e) =>
            setUploadProgress(e.lengthComputable ? e.loaded / e.total : 0);
          xhr.onerror = () =>
            reject(
              Error("Upload interrupted. Check network connection and storage availability."),
            );
          xhr.onload = () => {
            let data;
            try {
              data = JSON.parse(xhr.responseText);
            } catch {
              reject(Error("Upload failed. The server response was invalid."));
              return;
            }
            if (xhr.status === 413) {
              reject(Error("File exceeds the server maximum upload limit."));
            } else if (xhr.status === 507) {
              reject(Error(data.detail || "Insufficient server storage: disk space or storage quota exceeded."));
            } else if (xhr.status === 422) {
              reject(Error(data.detail || "Cannot decode recording. Ensure video is up to 4K 240 FPS / 10 minutes or upload a valid JPG/PNG."));
            } else if (xhr.status === 409) {
              reject(Error("Another upload or analysis is in progress. Try again shortly."));
            } else if (xhr.status >= 300) {
              reject(Error(data.detail || "Upload rejected by server storage limits."));
            } else {
              resolve(data);
            }
          };
          xhr.send(file);
        });
        await open(s);
        await refresh();
      } finally {
        setUploadProgress(null);
      }
    });

  const loadSampleVideo = () =>
    action(async () => {
      try {
        const s = await api("/sample", "POST");
        await open(s);
        await refresh();
      } catch (e) {
        throw Error((e as Error).message || "Could not load bundled sample video.");
      }
    });

  const list = sessions.filter(
    (s) =>
      s.file_name.toLowerCase().includes(search.toLowerCase()) &&
      (filter === "all" || s.status.state === filter),
  );

  const start = () =>
    action(async () => {
      if (!selected) return;
      await api(`/sessions/${selected.id}/start`, "POST", draft);
      await refresh();
    });

  const remove = (s: Session) =>
    action(async () => {
      if (!confirm(`Delete ${s.file_name} and its analysis permanently?`))
        return;
      await api("/sessions/" + s.id, "DELETE");
      if (selectedId.current === s.id) {
        selectedId.current = "";
        sessionStorage.removeItem("campus-session");
        setSelected(null);
      }
      await refresh();
    });

  if (auth === null)
    return (
      <div className="login">
        <Activity />
        <h1>Opening campus workspace</h1>
        {error && <p role="alert">{error}</p>}
      </div>
    );

  if (!auth)
    return (
      <main className="login">
        <div className="brand">
          <ScanLine /> CROWDVISION OPERATIONS
        </div>
        <h1>Recorded footage. Clearer decisions.</h1>
        <p>Sign in to your shared event-team workspace.</p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            action(async () => {
              await api("/login", "POST", { password });
              setPassword("");
              setAuth(true);
            });
          }}
        >
          <label>
            Team password
            <input
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <button className="primary" disabled={busy}>
            Open workspace <ArrowRight size={16} />
          </button>
        </form>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        <small>
          Private recordings · Authenticated access · {retention}-day retention
        </small>
      </main>
    );

  return (
    <div className="cv-app-shell">
      {/* 1. TOP HEADER (Brand Left, Operations Center, Sign Out Right) */}
      <header className="cv-header">
        <div className="cv-header-left">
          <div className="cv-brand-icon">
            <Users size={19} />
          </div>
          <div className="cv-brand-text">
            <span className="cv-brand-title">CrowdVision</span>
            <span className="cv-brand-subtitle">Campus Crowd Analytics</span>
          </div>
        </div>

        <div className="cv-header-center">
          <div className="cv-workspace-pill">
            <ScanLine size={15} />
            <span>Operations</span>
          </div>
        </div>

        <div className="cv-header-right">
          <button
            className="cv-signout-btn"
            aria-label="Sign out"
            onClick={() =>
              action(async () => {
                await api("/logout", "POST");
                setAuth(false);
              })
            }
          >
            <LogOut size={14} />
            <span>Sign Out</span>
          </button>
        </div>
      </header>

      {/* 2. SECONDARY HORIZONTAL SUB-NAV (4 Options + Footage Settings Dropdown) */}
      <div className="cv-subnav-wrap">
        <nav className="cv-subnav" aria-label="Secondary navigation">
          {/* Review Footage Navigation Button */}
          <button
            className={`cv-subnav-btn ${page === "Review Footage" ? "active" : ""}`}
            aria-current={page === "Review Footage" ? "page" : undefined}
            onClick={() => navigate("Review Footage")}
          >
            <Play size={16} />
            <span>Review Footage</span>
          </button>

          {/* Footage Settings dropdown placed directly beside Review Footage */}
          <div className="cv-footage-settings-anchor">
            <button
              className={`cv-footage-settings-btn ${footageSettingsOpen ? "active" : ""}`}
              aria-label="Recording and footage analysis settings"
              aria-expanded={footageSettingsOpen}
              onClick={() => setFootageSettingsOpen(!footageSettingsOpen)}
            >
              <SlidersHorizontal size={14} />
              <span>Footage Settings</span>
              <ChevronDown size={13} className={`cv-chevron ${footageSettingsOpen ? "open" : ""}`} />
            </button>

            {footageSettingsOpen && (
              <div
                className="cv-footage-settings-dropdown"
                role="dialog"
                aria-label="Footage settings"
              >
                <div className="cv-fs-header">
                  <h4>Footage Settings</h4>
                  <small>Default analysis parameters for recorded footage</small>
                </div>

                <div className="cv-fs-body">
                  <label>
                    Default Capacity (heads)
                    <input
                      type="number"
                      min="1"
                      max="10000"
                      value={draft.zones[0]?.capacity || 80}
                      onChange={(e) => {
                        const cap = Number(e.target.value) || 80;
                        const updated = {
                          ...draft,
                          zones: draft.zones.map((z, i) =>
                            i === 0 ? { ...z, capacity: cap } : z,
                          ),
                        };
                        setDraft(updated);
                        localStorage.setItem("campus-settings", JSON.stringify(updated));
                      }}
                    />
                  </label>

                  <label>
                    Buildup Duration (seconds)
                    <input
                      type="number"
                      min="1"
                      max="60"
                      step="0.5"
                      value={draft.event_seconds}
                      onChange={(e) => {
                        const sec = Number(e.target.value) || 3;
                        const updated = { ...draft, event_seconds: sec };
                        setDraft(updated);
                        localStorage.setItem("campus-settings", JSON.stringify(updated));
                      }}
                    />
                  </label>

                  <label>
                    Sampling FPS
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={draft.analysis_fps}
                      onChange={(e) => {
                        const fps = Number(e.target.value) || 5;
                        const updated = { ...draft, analysis_fps: fps };
                        setDraft(updated);
                        localStorage.setItem("campus-settings", JSON.stringify(updated));
                      }}
                    />
                  </label>

                  <label>
                    Confidence Threshold
                    <input
                      type="number"
                      min="0.05"
                      max="0.90"
                      step="0.05"
                      value={draft.conf}
                      onChange={(e) => {
                        const conf = Number(e.target.value) || 0.35;
                        const updated = { ...draft, conf };
                        setDraft(updated);
                        localStorage.setItem("campus-settings", JSON.stringify(updated));
                      }}
                    />
                  </label>

                  <label>
                    Output Resolution
                    <select
                      value={draft.imgsz}
                      onChange={(e) => {
                        const imgsz = Number(e.target.value) || 640;
                        const updated = { ...draft, imgsz };
                        setDraft(updated);
                        localStorage.setItem("campus-settings", JSON.stringify(updated));
                      }}
                    >
                      <option value="640">640px (Standard / High Detail)</option>
                      <option value="480">480px (Fast Processing)</option>
                      <option value="320">320px (Compact / Lightweight)</option>
                    </select>
                  </label>
                </div>

                <div className="cv-fs-footer">
                  <button
                    className="cv-btn-secondary"
                    onClick={() => {
                      setDraft(defaults);
                      localStorage.setItem("campus-settings", JSON.stringify(defaults));
                    }}
                  >
                    Reset Defaults
                  </button>
                  <button
                    className="cv-btn-primary"
                    onClick={() => setFootageSettingsOpen(false)}
                  >
                    Done
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Remaining 3 Navigation Tabs */}
          {[
            { id: "Live Monitoring" as const, label: "Live Monitoring", icon: Video },
            { id: "Activity & History" as const, label: "Activity & History", icon: Layers },
            { id: "Reports" as const, label: "Reports", icon: FileText },
          ].map((item) => {
            const Icon = item.icon;
            const isCurrent = page === item.id;
            return (
              <button
                key={item.id}
                className={`cv-subnav-btn ${isCurrent ? "active" : ""}`}
                aria-current={isCurrent ? "page" : undefined}
                onClick={() => navigate(item.id)}
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* 4. MAIN WORKSPACE CONTENT */}
      <main className="cv-main-content">
        {error && (
          <div className="error" role="alert">
            {error}
            <button aria-label="Dismiss error" onClick={() => setError("")}>
              <X size={16} />
            </button>
          </div>
        )}

        {/* SECTION 1: REVIEW FOOTAGE */}
        {page === "Review Footage" && (
          <>
            {!selected ? (
              /* Review Footage Empty State Matching Visual Reference */
              <div>
                <div className="cv-page-title-row">
                  <div className="cv-page-title-icon">
                    <Film size={20} />
                  </div>
                  <div>
                    <h1>Review Footage / Analysis</h1>
                    <p>Analyse recorded videos, define counting regions and monitor crowd occupancy and flow.</p>
                  </div>
                </div>

                {/* Upload Card */}
                <div className="cv-upload-card">
                  <div className="cv-upload-card-header">
                    <div className="cv-upload-card-icon">
                      <Upload size={18} />
                    </div>
                    <div>
                      <h2>Upload Recorded Footage</h2>
                      <p>Upload a campus video or image to begin crowd analysis.</p>
                    </div>
                  </div>

                  {/* Drag-and-drop Dropzone */}
                  <div
                    className={`cv-dropzone ${dragOver ? "dragover" : ""}`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOver(true);
                    }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragOver(false);
                      const file = e.dataTransfer.files?.[0];
                      if (file) upload(file);
                    }}
                  >
                    <div className="cv-dropzone-cloud">
                      <Upload size={38} />
                    </div>
                    <p className="cv-dropzone-title">Drag and drop video or image here</p>
                    <p className="cv-dropzone-subtitle">Supported formats: MP4, AVI, MOV, JPG, PNG (up to 4K, 10 min)</p>

                    <label className="cv-browse-btn">
                      <span>{busy ? "Uploading…" : "Browse Files"}</span>
                      <input
                        type="file"
                        accept=".mp4,.mov,.avi,.mkv,.jpg,.jpeg,.png"
                        disabled={busy}
                        style={{ display: "none" }}
                        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
                      />
                    </label>

                    {uploadProgress !== null && (
                      <div className="cv-upload-progress-wrap">
                        <div className="cv-progress-bar-bg">
                          <div
                            className="cv-progress-bar-fill"
                            style={{ width: `${Math.round(uploadProgress * 100)}%` }}
                          />
                        </div>
                        <span>{Math.round(uploadProgress * 100)}%</span>
                      </div>
                    )}
                  </div>

                  {/* Footer Row */}
                  <div className="cv-upload-footer">
                    <button
                      className="cv-sample-btn"
                      disabled={busy}
                      onClick={loadSampleVideo}
                    >
                      <FolderOpen size={16} />
                      <span>Load Sample Video</span>
                    </button>

                    <div className="cv-privacy-indicator">
                      <Lock size={14} />
                      <span>
                        {systemStatus?.deployment_mode === "hosted" ? (
                          <>
                            <strong>Server Processing</strong> · Video inference runs on the secure application server.
                          </>
                        ) : (
                          <>
                            <strong>Local Processing</strong> · Video inference runs locally on your machine.
                          </>
                        )}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Capabilities Section */}
                <div className="cv-capabilities-header">
                  <Layers size={17} />
                  <h3>CrowdVision Capabilities</h3>
                </div>

                <div className="cv-capabilities-grid">
                  <div className="cv-capability-card">
                    <div className="cv-cap-icon emerald">
                      <Users size={22} />
                    </div>
                    <h4>Crowd Detection & Occupancy</h4>
                    <p>YOLO11-based people detection and occupancy counting.</p>
                  </div>

                  <div className="cv-capability-card">
                    <div className="cv-cap-icon copper">
                      <Activity size={22} />
                    </div>
                    <h4>Directional Flow Analysis</h4>
                    <p>Track movement and calculate entry and exit flow.</p>
                  </div>

                  <div className="cv-capability-card">
                    <div className="cv-cap-icon amber">
                      <AlertTriangle size={22} />
                    </div>
                    <h4>Safety & Incident Detection</h4>
                    <p>Monitor capacity limits and detect sustained crowding.</p>
                  </div>
                </div>
              </div>
            ) : (
              /* Video Selected: Zone Editor & Analysis Workflow */
              <div>
                <div className="recording-bar">
                  <Film size={18} />
                  <b>{selected.file_name}</b>
                  <span>
                    {fmt(selected.duration)} · {mb(selected.bytes)}
                  </span>
                  <Badge state={selected.status.state} />
                  <button
                    className="link"
                    style={{ marginLeft: "auto" }}
                    onClick={() => {
                      selectedId.current = "";
                      sessionStorage.removeItem("campus-session");
                      setSelected(null);
                    }}
                  >
                    Change recording
                  </button>
                </div>

                {selected.result ? (
                  <Results key={selected.id} session={selected} />
                ) : active(selected) ? (
                  <div className="processing">
                    <div>
                      <img
                        src={`/api/sessions/${selected.id}/media/poster`}
                        alt="Recording preview"
                      />
                    </div>
                    <section>
                      <Activity size={24} />
                      <h2>Analyzing your recording</h2>
                      <p>
                        {selected.status.message || "Loading the detector…"}
                      </p>
                      <progress
                        value={selected.status.progress || 0}
                        max={1}
                      />
                      <p className="quiet">
                        You can navigate away or close this page. Processing
                        continues on the server.
                      </p>
                      <button
                        disabled={busy}
                        onClick={() =>
                          action(async () => {
                            await api(
                              `/sessions/${selected.id}/cancel`,
                              "POST",
                            );
                            await refresh();
                          })
                        }
                      >
                        Cancel analysis
                      </button>
                      <details>
                        <summary>Settings used for this job</summary>
                        <pre>
                          {JSON.stringify(selected.settings, null, 2)}
                        </pre>
                      </details>
                    </section>
                  </div>
                ) : (
                  <>
                    {selected.status.error && (
                      <div className="error" role="alert">
                        {selected.status.error}
                      </div>
                    )}
                    {selected.settings?.zones && (
                      <div className="notice">
                        <Settings2 size={18} />
                        <div>
                          <b>Editing your current draft settings</b>
                          <p>
                            The previous attempt used its own saved
                            configuration.
                          </p>
                          <button
                            onClick={() => setDraft(selected.settings!)}
                          >
                            Restore settings from previous attempt
                          </button>
                        </div>
                      </div>
                    )}
                    <ZoneEditor
                      session={selected}
                      settings={draft}
                      setSettings={setDraft}
                    />
                    <div className="start-bar">
                      <p>
                        <b>Ready to analyze?</b>
                        <br />
                        <span>
                          One detector pass per sampled frame. Outputs are
                          compressed and storage is capped.
                        </span>
                      </p>
                      <button
                        className="primary"
                        disabled={busy}
                        onClick={start}
                      >
                        Start analysis <ArrowRight size={17} />
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}
          </>
        )}

        {/* SECTION 2: LIVE MONITORING */}
        {page === "Live Monitoring" && (
          <div>
            <div className="cv-page-title-row">
              <div className="cv-page-title-icon">
                <Video size={20} />
              </div>
              <div>
                <h1>Live Monitoring & RTSP Surveillance</h1>
                <p>Connect IP surveillance cameras or local devices for real-time occupancy monitoring.</p>
              </div>
            </div>

            <div className="cv-live-grid">
              <div className="panel">
                <h2>Stream Configuration</h2>
                <label>
                  Camera Feed Mode
                  <select
                    value={liveSourceType}
                    onChange={(e) => setLiveSourceType(e.target.value)}
                  >
                    <option value="rtsp">RTSP / Network IP Camera</option>
                    <option value="webcam">Local USB / Integrated Webcam</option>
                  </select>
                </label>

                {liveSourceType === "rtsp" ? (
                  <label>
                    RTSP Stream Address
                    <input
                      type="text"
                      placeholder="rtsp://camera.campus.internal:554/live"
                      value={rtspUrl}
                      onChange={(e) => {
                        setRtspUrl(e.target.value);
                        localStorage.setItem("campus-rtsp-url", e.target.value);
                      }}
                    />
                  </label>
                ) : (
                  <label>
                    Local Camera Device
                    <select
                      value={rtspUrl}
                      onChange={(e) => {
                        setRtspUrl(e.target.value);
                        localStorage.setItem("campus-rtsp-url", e.target.value);
                      }}
                    >
                      <option value="0">Camera 0 (Default Integrated)</option>
                      <option value="1">Camera 1 (External USB)</option>
                    </select>
                  </label>
                )}

                <div className="cv-grid-2">
                  <label>
                    Monitored Location
                    <input
                      type="text"
                      value={liveLocation}
                      onChange={(e) => setLiveLocation(e.target.value)}
                    />
                  </label>
                  <label>
                    Capacity Threshold (heads)
                    <input
                      type="number"
                      min="1"
                      max="10000"
                      value={liveCapacity}
                      onChange={(e) => setLiveCapacity(Number(e.target.value))}
                    />
                  </label>
                </div>

                <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                  <button
                    className="cv-btn-primary"
                    disabled={cameraCheck.checking}
                    onClick={testCameraConnection}
                  >
                    {cameraCheck.checking ? "Testing Stream..." : "Test Stream Connection"}
                  </button>
                </div>

                {/* Honest backend camera connection status */}
                <div className="cv-camera-status-card" style={{ marginTop: 14 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span
                      className={`cv-status-dot ${cameraCheck.connected ? "connected" : cameraCheck.checking ? "warning" : ""}`}
                      style={{
                        background: cameraCheck.connected
                          ? "var(--success)"
                          : cameraCheck.checking
                          ? "var(--warning)"
                          : "var(--text-muted)",
                        boxShadow: cameraCheck.connected ? "0 0 8px var(--success)" : "none",
                      }}
                    />
                    <strong
                      style={{
                        fontSize: 13,
                        color: cameraCheck.connected ? "var(--success)" : "var(--text-primary)",
                      }}
                    >
                      {cameraCheck.connected
                        ? `Connected (${cameraCheck.width}×${cameraCheck.height} @ ${cameraCheck.fps} FPS)`
                        : cameraCheck.checking
                        ? "Probing Stream..."
                        : cameraCheck.tested
                        ? "Connection Failed"
                        : "Not Connected"}
                    </strong>
                  </div>
                  <p className="hint" style={{ marginTop: 6, marginBottom: 0 }}>
                    {cameraCheck.message}
                  </p>
                </div>
              </div>

              <div className="panel">
                <h2>Live Surveillance Monitor</h2>
                <div className="cv-live-stage">
                  {cameraCheck.connected ? (
                    <div className="cv-live-placeholder active">
                      <span className="cv-live-rec-badge">● STREAM VERIFIED & READY</span>
                      <p style={{ marginTop: 16 }}>
                        Verified live input from {liveSourceType === "rtsp" ? rtspUrl : "Local Camera Device"}
                      </p>
                      <small>
                        Resolution: {cameraCheck.width}×{cameraCheck.height} · Framerate: {cameraCheck.fps} FPS
                      </small>
                      <p style={{ marginTop: 8, fontSize: 12, color: "var(--secondary)" }}>
                        Active zone: {liveLocation} · Threshold: {liveCapacity} heads
                      </p>
                    </div>
                  ) : (
                    <div className="cv-live-placeholder">
                      <VideoOff size={36} style={{ color: "var(--text-muted)", marginBottom: 12 }} />
                      <p style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: 4 }}>
                        Camera Not Connected
                      </p>
                      <p style={{ fontSize: 12, color: "var(--text-secondary)", maxWidth: 360, margin: "0 auto" }}>
                        Configure and test your RTSP stream address or local device on the left. The system requires an authenticated, accessible video source and will not simulate an unverified live feed.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* SECTION 3: ACTIVITY & HISTORY */}
        {page === "Activity & History" && (
          <div>
            <div className="cv-page-title-row">
              <div className="cv-page-title-icon">
                <FolderOpen size={20} />
              </div>
              <div>
                <h1>Activity & History</h1>
                <p>Saved analysis sessions, cross-zone comparative telemetry, and safety incident audit log.</p>
              </div>
            </div>

            <div className="stats">
              <Stat
                label="Saved recordings"
                value={sessions.length}
                note={`Retained for ${retention} days`}
              />
              <Stat
                label="Completed analyses"
                value={sessions.filter((s) => s.status.state === "complete").length}
                note="Ready for review & export"
              />
              <Stat
                label="Inference worker"
                value={sessions.some(active) ? "Processing" : "Idle"}
                note="PyTorch Direct CPU / CUDA"
              />
            </div>

            <div className="filters">
              <label className="search">
                <Search size={17} />
                <input
                  aria-label="Search recordings"
                  placeholder="Search recordings by name…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </label>
              <select
                aria-label="Filter sessions by status"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              >
                {["all", "ready", "running", "complete", "failed", "cancelled"].map((v) => (
                  <option key={v} value={v}>
                    {v === "all" ? "All statuses" : v}
                  </option>
                ))}
              </select>
              <span className="quiet">{list.length} recordings</span>
            </div>

            {list.length ? (
              <SessionList
                sessions={list}
                open={(s) => action(() => open(s))}
                remove={remove}
              />
            ) : (
              <Empty
                icon={<FolderOpen />}
                title="No recordings found"
                text="Try a different search or upload a recording to start a review."
              />
            )}
          </div>
        )}

        {/* SECTION 4: REPORTS */}
        {page === "Reports" && (
          <div>
            <div className="cv-page-title-row">
              <div className="cv-page-title-icon">
                <FileText size={20} />
              </div>
              <div>
                <h1>Compliance Reports & Data Exports</h1>
                <p>Generate PDF inspection certificates and export timestamped counting telemetry.</p>
              </div>
            </div>

            {sessions.filter((s) => s.status.state === "complete").length ? (
              <div className="cv-reports-grid">
                {sessions
                  .filter((s) => s.status.state === "complete")
                  .map((s) => (
                    <div className="cv-report-card" key={s.id}>
                      <div className="cv-report-card-top">
                        <img src={`/api/sessions/${s.id}/media/poster`} alt="" />
                        <div>
                          <b>{s.file_name}</b>
                          <p>
                            {new Date(s.created * 1000).toLocaleString()} · {fmt(s.duration)} · {mb(s.bytes)}
                          </p>
                          <span className="badge complete" style={{ display: "inline-block", marginTop: 4 }}>
                            Completed Analysis
                          </span>
                        </div>
                      </div>

                      <div className="cv-report-actions">
                        <a
                          href={`/api/sessions/${s.id}/export/pdf`}
                          download
                          className="cv-export-link"
                        >
                          <Download size={14} /> PDF Report
                        </a>
                        <a
                          href={`/api/sessions/${s.id}/export/csv`}
                          download
                          className="cv-export-link"
                        >
                          <Download size={14} /> Counts CSV
                        </a>
                        <a
                          href={`/api/sessions/${s.id}/export/events.csv`}
                          download
                          className="cv-export-link"
                        >
                          <Download size={14} /> Events CSV
                        </a>
                        <a
                          href={`/api/sessions/${s.id}/export/mp4`}
                          download
                          className="cv-export-link"
                        >
                          <Download size={14} /> Video MP4
                        </a>
                        <button
                          className="cv-btn-secondary"
                          style={{ marginLeft: "auto" }}
                          onClick={() => action(() => open(s))}
                        >
                          Open Footage <ArrowRight size={14} />
                        </button>
                      </div>
                    </div>
                  ))}
              </div>
            ) : (
              <Empty
                icon={<FileText />}
                title="No completed reports yet"
                text="Upload a recording and complete an analysis to generate downloadable PDF inspection certificates and telemetry datasets."
                action={
                  <button className="primary" onClick={() => navigate("Review Footage")}>
                    Review Footage <ArrowRight size={16} />
                  </button>
                }
              />
            )}
          </div>
        )}
      </main>
    </div>
  );
}

function Stat({
  label,
  value,
  note,
}: {
  label: string;
  value: React.ReactNode;
  note: string;
}) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

function Badge({ state }: { state: string }) {
  return (
    <span className={"badge " + state}>
      {state === "complete"
        ? "Completed"
        : state === "ready"
          ? "Ready to configure"
          : state}
    </span>
  );
}

function Empty({
  icon,
  title,
  text,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="empty">
      {icon}
      <h2>{title}</h2>
      <p>{text}</p>
      {action}
    </div>
  );
}

function SessionList({
  sessions,
  open,
  remove,
}: {
  sessions: Session[];
  open: (s: Session) => void;
  remove: (s: Session) => void;
}) {
  return (
    <div className="session-list">
      {sessions.map((s) => (
        <div className="session-row" key={s.id}>
          <img src={`/api/sessions/${s.id}/media/poster`} alt="" />
          <button className="session-name" onClick={() => open(s)}>
            <b>{s.file_name}</b>
            <span>
              {new Date(s.created * 1000).toLocaleString()} · {fmt(s.duration)}{" "}
              · {mb(s.bytes)}
            </span>
          </button>
          <Badge state={s.status.state} />
          <button aria-label={"Open " + s.file_name} onClick={() => open(s)}>
            <ArrowRight size={18} />
          </button>
          <button
            disabled={active(s)}
            aria-label={"Delete " + s.file_name}
            onClick={() => remove(s)}
          >
            <Trash2 size={16} />
          </button>
        </div>
      ))}
    </div>
  );
}

function ZoneEditor({
  session,
  settings,
  setSettings,
}: {
  session: Session;
  settings: Settings;
  setSettings: (s: Settings) => void;
}) {
  const [zone, setZone] = useState(0),
    [drawing, setDrawing] = useState(false),
    [points, setPoints] = useState<Point[]>([]);
  const index = Math.min(zone, settings.zones.length - 1),
    current = settings.zones[index];
  const update = (patch: Partial<Zone>) =>
    setSettings({
      ...settings,
      zones: settings.zones.map((z, i) =>
        i === index ? { ...z, ...patch } : z,
      ),
    });
  const apply = () => {
    if (points.length >= 3) {
      update({ polygon: points });
      setDrawing(false);
      setPoints([]);
    }
  };
  return (
    <div className="editor-layout">
      <section className="panel scene-panel">
        <div className="section-title">
          <h2>Counting areas</h2>
          <span className="quiet">{settings.zones.length} / 5 zones</span>
        </div>
        <div
          className="scene"
          tabIndex={drawing ? 0 : -1}
          role="group"
          aria-label="Counting zone editor"
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              setDrawing(false);
              setPoints([]);
            }
            if (e.key === "Enter") apply();
          }}
        >
          <img
            src={`/api/sessions/${session.id}/media/poster`}
            alt="First frame of uploaded recording"
          />
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            onClick={(e) => {
              if (!drawing || points.length >= 12) return;
              const r = e.currentTarget.getBoundingClientRect();
              setPoints([
                ...points,
                [
                  (e.clientX - r.left) / r.width,
                  (e.clientY - r.top) / r.height,
                ],
              ]);
            }}
          >
            {settings.zones.map((z, i) => (
              <polygon
                key={i}
                points={z.polygon
                  .map((p) => p.map((v) => v * 100).join(","))
                  .join(" ")}
                className={i === index ? "zone-active" : "zone-other"}
              />
            ))}
            {drawing && (
              <polyline
                points={points
                  .map((p) => p.map((v) => v * 100).join(","))
                  .join(" ")}
                fill="rgba(8, 127, 91, 0.25)"
                stroke="#087F5B"
                strokeWidth=".6"
              />
            )}
            {points.map((p, i) => (
              <circle
                key={i}
                cx={p[0] * 100}
                cy={p[1] * 100}
                r=".8"
                fill="#087F5B"
              />
            ))}
          </svg>
          <span className="scene-label">
            {drawing ? `Drawing · ${points.length} corners` : current.name}
          </span>
        </div>
        <div className="scene-tools">
          <button
            onClick={() => {
              setDrawing(!drawing);
              setPoints([]);
            }}
          >
            <ScanLine size={16} />
            {drawing ? "Cancel drawing" : "Draw polygon"}
          </button>
          {drawing && (
            <>
              <button disabled={points.length < 3} onClick={apply}>
                Apply zone
              </button>
              <button
                disabled={!points.length}
                onClick={() => setPoints(points.slice(0, -1))}
              >
                Undo corner
              </button>
            </>
          )}
          <button
            onClick={() => {
              update({ polygon: defaults.zones[0].polygon });
              setDrawing(false);
            }}
          >
            Use full frame
          </button>
        </div>
        <p className="hint" style={{ marginTop: 8 }}>
          Click 3–12 corners clockwise to draw a convex area. Overlapping zones
          share detections; overall counts do not double-count them.
        </p>
        <details style={{ marginTop: 8 }}>
          <summary>Keyboard-accessible zone coordinates</summary>
          <p className="hint">
            Normalized coordinates: 0 is left/top; 1 is right/bottom. Edit each
            corner without drawing.
          </p>
          {current.polygon.map((p, i) => (
            <div className="coordinate" key={i} style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 6 }}>
              <span>Corner {i + 1}</span>
              {[0, 1].map((axis) => (
                <label key={axis} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  {axis ? "Y" : "X"}
                  <input
                    aria-label={`Corner ${i + 1} ${axis ? "Y" : "X"}`}
                    type="number"
                    min="0"
                    max="1"
                    step=".01"
                    value={p[axis]}
                    style={{ width: 70 }}
                    onChange={(e) =>
                      update({
                        polygon: current.polygon.map((v, j) =>
                          j === i
                            ? (v.map((n, a) =>
                                a === axis ? Number(e.target.value) : n,
                              ) as Point)
                            : v,
                        ),
                      })
                    }
                  />
                </label>
              ))}
            </div>
          ))}
        </details>
      </section>

      <section className="panel config-panel">
        <h2>Analysis configuration</h2>
        <label>
          Location / event
          <input
            maxLength={80}
            value={settings.location}
            onChange={(e) =>
              setSettings({ ...settings, location: e.target.value })
            }
          />
        </label>
        <div className="section-title">
          <h3>Named zones</h3>
          <button
            aria-label="Add zone"
            disabled={settings.zones.length >= 5}
            onClick={() => {
              setSettings({
                ...settings,
                zones: [
                  ...settings.zones,
                  {
                    ...defaults.zones[0],
                    name: "Zone " + (settings.zones.length + 1),
                  },
                ],
              });
              setZone(settings.zones.length);
            }}
          >
            <Plus size={16} /> Add zone
          </button>
        </div>
        <div className="zone-tabs">
          {settings.zones.map((z, i) => (
            <button
              key={i}
              aria-pressed={i === index}
              className={i === index ? "chosen" : ""}
              onClick={() => {
                setZone(i);
                setDrawing(false);
              }}
            >
              {i + 1}. {z.name}
            </button>
          ))}
        </div>
        <label>
          Zone name
          <input
            maxLength={40}
            value={current.name}
            onChange={(e) => update({ name: e.target.value })}
          />
        </label>
        <label>
          Zone capacity · heads
          <input
            type="number"
            min="1"
            max="10000"
            value={current.capacity}
            onChange={(e) => update({ capacity: Number(e.target.value) })}
          />
        </label>
        <p className="hint">
          A review threshold set by your team, not a certified safe capacity.
        </p>
        {settings.zones.length > 1 && (
          <button
            className="link danger"
            onClick={() => {
              setSettings({
                ...settings,
                zones: settings.zones.filter((_, i) => i !== index),
              });
              setZone(0);
            }}
          >
            Remove selected zone
          </button>
        )}
        <label>
          Minimum buildup duration · seconds
          <input
            type="number"
            min="1"
            max="60"
            value={settings.event_seconds}
            onChange={(e) =>
              setSettings({
                ...settings,
                event_seconds: Number(e.target.value),
              })
            }
          />
        </label>
        <details>
          <summary>
            <Settings2 size={15} /> Advanced detector settings
          </summary>
          <label>
            Confidence threshold
            <input
              type="number"
              step=".05"
              min=".05"
              max=".9"
              value={settings.conf}
              onChange={(e) =>
                setSettings({ ...settings, conf: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Analysis samples / second
            <input
              type="number"
              min="1"
              max="10"
              value={settings.analysis_fps}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  analysis_fps: Number(e.target.value),
                })
              }
            />
          </label>
          <p className="hint">
            The evaluation used confidence 0.35 and image size 640. Changing
            settings can change counting performance.
          </p>
        </details>
        <small className="saved" style={{ display: "inline-flex", alignItems: "center", gap: 4, marginTop: 8 }}>
          <Check size={13} style={{ color: "var(--success)" }} /> Draft settings saved on this device
        </small>
      </section>
    </div>
  );
}

function Results({ session }: { session: Session }) {
  const r = session.result!,
    video = useRef<HTMLVideoElement>(null),
    [time, setTime] = useState(0),
    [zone, setZone] = useState(-1),
    [heat, setHeat] = useState(false),
    [eventIndex, setEventIndex] = useState(-1);
  const times = r.timestamps,
    counts = zone < 0 ? r.per_frame_counts : r.zones[zone].counts,
    events = r.events.filter((e) => zone < 0 || e.zone === r.zones[zone].name),
    duration = Math.max(session.duration, times[times.length - 1] || 1),
    max = Math.max(1, ...counts, zone >= 0 ? r.zones[zone].capacity : 0);
  const seek = (t: number) => {
    if (video.current) {
      video.current.currentTime = t;
      setTime(t);
    }
  };
  const heatmax = Math.max(1, ...r.heatmap.grid.flat());
  return (
    <>
      <div className="result-toolbar">
        <span className="quiet" style={{ color: "var(--text-primary)", fontWeight: 600 }}>
          <Check size={16} style={{ color: "var(--success)" }} /> Completed · {r.settings.location}
        </span>
        <details className="export">
          <summary>
            <Download size={16} /> Export results
          </summary>
          <div>
            {[
              ["pdf", "Analysis report (.pdf)"],
              ["events.csv", "Buildup events (.csv)"],
              ["csv", "Sampled counts (.csv)"],
              ["json", "Zones, events & settings (.json)"],
              ["mp4", "Annotated footage (.mp4)"],
            ].map(([k, v]) => (
              <a
                key={k}
                href={`/api/sessions/${session.id}/export/${k}`}
                download
              >
                {v}
              </a>
            ))}
          </div>
        </details>
      </div>

      <div className="stats">
        <Stat
          label="Peak observed occupancy"
          value={Math.max(...counts) + " heads"}
          note={
            zone < 0 ? "Union of all configured zones" : "Selected zone only"
          }
        />
        <Stat
          label="Average observed occupancy"
          value={
            (counts.reduce((a, b) => a + b, 0) / counts.length).toFixed(1) +
            " heads"
          }
          note="Mean across sampled frames"
        />
        <Stat
          label="Sustained buildup events"
          value={events.length}
          note={`Above zone capacity for ≥ ${r.settings.event_seconds} seconds`}
        />
      </div>

      <div className="results-layout">
        <section className="panel footage">
          <div className="section-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <h2>Footage review</h2>
            <button aria-pressed={heat} onClick={() => setHeat(!heat)}>
              {heat ? "Hide heatmap" : "Show activity heatmap"}
            </button>
          </div>
          <div className="video-stage">
            {session.media_type === "image" || session.file_name.match(/\.(jpg|jpeg|png)$/i) ? (
              <img
                src={`/api/sessions/${session.id}/media/video`}
                alt="Annotated detection review"
                className="cv-annotated-image"
              />
            ) : (
              <video
                ref={video}
                controls
                playsInline
                preload="metadata"
                src={`/api/sessions/${session.id}/media/video`}
                onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
              />
            )}
            {heat && r.heatmap?.grid && (
              <svg
                viewBox="0 0 64 36"
                preserveAspectRatio="none"
                aria-label="Detection-frequency heatmap"
              >
                {r.heatmap.grid.flatMap((row, y) =>
                  row.map((v, x) =>
                    v > 0 ? (
                      <rect
                        key={y * 64 + x}
                        x={x}
                        y={y}
                        width="1"
                        height="1"
                        fill="#D9A441"
                        opacity={0.12 + (0.7 * v) / heatmax}
                      />
                    ) : null,
                  ),
                )}
              </svg>
            )}
          </div>
          {heat && (
            <p className="hint heat-legend" style={{ marginTop: 6 }}>
              Low <span style={{ display: "inline-block", width: 40, height: 6, background: "linear-gradient(90deg, rgba(217, 164, 65, 0.15), #D9A441)" }} /> High · Relative detection frequency across{" "}
              {r.heatmap.samples} samples. Not physical density or unique
              people.
            </p>
          )}
          <div className="timeline-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 14 }}>
            <h3>Occupancy timeline</h3>
            <label className="inline-label" style={{ fontSize: 12 }}>
              Counting area{" "}
              <select
                aria-label="Timeline counting area"
                value={zone}
                onChange={(e) => {
                  setZone(Number(e.target.value));
                  setEventIndex(-1);
                }}
              >
                <option value={-1}>All zones · union</option>
                {r.zones.map((z, i) => (
                  <option key={i} value={i}>
                    {z.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="chart">
            <div className="chart-scale">
              <span>{max} heads</span>
              <span>0</span>
            </div>
            <svg
              viewBox="0 0 800 150"
              preserveAspectRatio="none"
              aria-label="Sampled occupancy timeline"
              onClick={(e) => {
                const b = e.currentTarget.getBoundingClientRect();
                seek(((e.clientX - b.left) / b.width) * duration);
              }}
            >
              {[0, 50, 100, 149].map((y) => (
                <line key={y} x1="0" x2="800" y1={y} y2={y} stroke="#454B43" />
              ))}
              {events.map((e, i) => (
                <rect
                  key={i}
                  x={(e.start_seconds / duration) * 800}
                  y="0"
                  width={Math.max(2, (e.duration_seconds / duration) * 800)}
                  height="150"
                  fill="rgba(217, 164, 65, 0.25)"
                />
              ))}
              {zone >= 0 && (
                <line
                  x1="0"
                  x2="800"
                  y1={145 - (r.zones[zone].capacity / max) * 135}
                  y2={145 - (r.zones[zone].capacity / max) * 135}
                  stroke="#D98236"
                  strokeDasharray="5 5"
                />
              )}
              <polyline
                fill="none"
                stroke="#087F5B"
                strokeWidth="2.5"
                points={times
                  .map(
                    (t, i) =>
                      `${(t / duration) * 800},${145 - (counts[i] / max) * 135}`,
                  )
                  .join(" ")}
              />
              <line
                x1={(time / duration) * 800}
                x2={(time / duration) * 800}
                y1="0"
                y2="150"
                stroke="#B87345"
                strokeWidth="2"
              />
            </svg>
          </div>
          <input
            className="scrubber"
            aria-label="Seek recording time"
            type="range"
            min="0"
            max={duration}
            step=".1"
            value={time}
            onChange={(e) => seek(Number(e.target.value))}
          />
          <div className="time-labels" style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, color: "var(--text-muted)" }}>
            <span>
              {fmt(time)} / {fmt(duration)}
            </span>
            <span>
              Emerald: observed heads · Amber: buildup events
              {zone >= 0 ? " · Dashed: capacity" : ""}
            </span>
          </div>
        </section>

        <section className="panel event-panel">
          <div className="section-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2>Review events</h2>
            <span className="count-label" style={{ background: "var(--surface-elevated)", padding: "2px 8px", borderRadius: 10, fontSize: 11 }}>{events.length}</span>
          </div>
          <p className="hint">
            Select an event to seek to its first observed over-capacity sample.
          </p>
          {events.length ? (
            events.map((e, i) => (
              <button
                key={i}
                className={
                  "event " + (eventIndex === i ? "event-selected" : "")
                }
                onClick={() => {
                  setEventIndex(i);
                  seek(e.start_seconds);
                }}
                aria-pressed={eventIndex === i}
              >
                <span className="event-top">
                  <b>{e.zone}</b>
                  <ArrowRight size={15} />
                </span>
                <span>
                  <Clock3 size={13} />
                  {fmt(e.start_seconds)}–{fmt(e.end_seconds)}
                </span>
                <strong>
                  {e.peak_count} heads <small>/ {e.capacity} capacity</small>
                </strong>
                <span>{e.duration_seconds.toFixed(1)} seconds sustained</span>
              </button>
            ))
          ) : (
            <div className="small-empty" style={{ textAlign: "center", padding: "30px 10px", color: "var(--text-muted)" }}>
              <Check size={22} style={{ color: "var(--success)", marginBottom: 8 }} />
              <h3>No sustained buildup detected</h3>
              <p style={{ fontSize: 12 }}>
                No sampled episode met this zone's capacity and duration
                thresholds.
              </p>
            </div>
          )}
        </section>
      </div>

      <details className="panel">
        <summary style={{ cursor: "pointer", fontWeight: 600 }}>Analysis record & settings used</summary>
        <p style={{ marginTop: 8 }}>
          These settings belong to this completed result. Your current draft
          settings are separate.
        </p>
        <div className="record-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, margin: "14px 0" }}>
          <span style={{ display: "flex", flexDirection: "column", fontSize: 12, color: "var(--text-muted)" }}>
            Sampling<b style={{ color: "var(--text-primary)", fontSize: 13 }}>{r.settings.analysis_fps} frames / sec</b>
          </span>
          <span style={{ display: "flex", flexDirection: "column", fontSize: 12, color: "var(--text-muted)" }}>
            Confidence<b style={{ color: "var(--text-primary)", fontSize: 13 }}>{r.settings.conf}</b>
          </span>
          <span style={{ display: "flex", flexDirection: "column", fontSize: 12, color: "var(--text-muted)" }}>
            Analyzed frames<b style={{ color: "var(--text-primary)", fontSize: 13 }}>{r.processed_frames}</b>
          </span>
          <span style={{ display: "flex", flexDirection: "column", fontSize: 12, color: "var(--text-muted)" }}>
            Output size
            <b style={{ color: "var(--text-primary)", fontSize: 13 }}>
              {mb(r.output_bytes)} / {mb(r.input_bytes)} input
            </b>
          </span>
        </div>
        {r.zones.map((z) => (
          <p key={z.name} style={{ margin: "4px 0", fontSize: 12.5 }}>
            <b>{z.name}</b> · capacity {z.capacity} heads · observed peak{" "}
            {z.peak} heads
          </p>
        ))}
        <p className="hint" style={{ marginTop: 10 }}>
          Counts may miss occluded heads or include false detections. Heatmap
          cells accumulate head-center observations; perspective is not
          calibrated. This is recorded-footage review, not a safety prediction.
        </p>
      </details>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
