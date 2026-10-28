import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Check,
  ChevronRight,
  Clock3,
  Download,
  Film,
  FolderOpen,
  LayoutDashboard,
  LogOut,
  Plus,
  ScanLine,
  Search,
  Settings2,
  ShieldCheck,
  Trash2,
  Upload,
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
const pages = ["Overview", "Analyze footage", "Sessions", "Model evaluation"];
const icons = [LayoutDashboard, ScanLine, FolderOpen, BarChart3];
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
function App() {
  const [page, setPage] = useState(
    sessionStorage.getItem("campus-page") || "Overview",
  );
  const [auth, setAuth] = useState<boolean | null>(null),
    [password, setPassword] = useState(""),
    [retention, setRetention] = useState(7);
  const [sessions, setSessions] = useState<Session[]>([]),
    [selected, setSelected] = useState<Session | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [search, setSearch] = useState(""),
    [filter, setFilter] = useState("all");
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
  const [uploadProgress, setUploadProgress] = useState<number | null>(null),
    [evaluation, setEvaluation] = useState<any>(null);
  const selectedId = useRef(sessionStorage.getItem("campus-session") || "");
  const navigate = (p: string) => {
    setPage(p);
    sessionStorage.setItem("campus-page", p);
    window.scrollTo(0, 0);
  };
  const refresh = async () => {
    const s = await api("/sessions");
    setSessions(s);
    if (selectedId.current) {
      try {
        setSelected(await api("/sessions/" + selectedId.current));
      } catch {
        selectedId.current = "";
        setSelected(null);
      }
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
    api("/evaluation")
      .then(setEvaluation)
      .catch((e) => setError(e.message));
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
    navigate("Analyze footage");
  };
  const upload = (file: File) =>
    action(async () => {
      if (file.size > 200 * 1024 ** 2)
        throw Error("Choose a recording smaller than 200 MB.");
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
              Error("Upload interrupted. Check your connection and try again."),
            );
          xhr.onload = () => {
            let data;
            try {
              data = JSON.parse(xhr.responseText);
            } catch {
              reject(Error("Upload failed. The server may be unavailable."));
              return;
            }
            xhr.status < 300
              ? resolve(data)
              : reject(Error(data.detail || "Upload failed."));
          };
          xhr.send(file);
        });
        await open(s);
        await refresh();
      } finally {
        setUploadProgress(null);
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
          <ScanLine /> CAMPUS / VISION
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
    <div className="shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("Overview");
          }}
        >
          <ScanLine size={25} />
          <span>
            CAMPUS<span className="brand-sub">VISION WORKSPACE</span>
          </span>
        </a>
        <div className="nav-label">RECORDED FOOTAGE</div>
        <nav aria-label="Main navigation">
          {pages.map((p, i) => {
            const Icon = icons[i];
            return (
              <button
                key={p}
                className={page === p ? "selected" : ""}
                aria-current={page === p ? "page" : undefined}
                onClick={() => navigate(p)}
              >
                <Icon size={18} />
                {p}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <span className="status-dot" /> CPU analysis worker
          <p>
            One recording at a time
            <br />
            Shared campus team workspace
          </p>
          <button
            onClick={() =>
              action(async () => {
                await api("/logout", "POST");
                setAuth(false);
              })
            }
          >
            <LogOut size={16} />
            Sign out
          </button>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <span>
            Campus operations <ChevronRight size={14} /> {page}
          </span>
          <span className="quiet">
            <ShieldCheck size={15} /> Private workspace
          </span>
          <button
            className="mobile-signout"
            aria-label="Sign out of workspace"
            onClick={() =>
              action(async () => {
                await api("/logout", "POST");
                setAuth(false);
              })
            }
          >
            <LogOut size={15} />
          </button>
        </header>
        <div className="content">
          {error && (
            <div className="error" role="alert">
              {error}
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          <div className="page-heading">
            <div>
              <p className="eyebrow">CROWD OCCUPANCY & REVIEW</p>
              <h1>
                {page === "Analyze footage" && selected?.result
                  ? "Analysis workspace"
                  : page}
              </h1>
              <p>
                {page === "Overview"
                  ? "Review activity, identify sustained buildup, and return to the moments that matter."
                  : page === "Analyze footage"
                    ? "Define the areas that matter. Review observed head counts against your configured capacities."
                    : page === "Sessions"
                      ? `Recordings and results stay available for ${retention} days. Search, revisit or remove a session.`
                      : "Measured performance, dataset provenance and the limits of this detector."}
              </p>
            </div>
            {page !== "Analyze footage" && (
              <button
                className="primary"
                onClick={() => {
                  selectedId.current = "";
                  sessionStorage.removeItem("campus-session");
                  setSelected(null);
                  navigate("Analyze footage");
                }}
              >
                <Plus size={17} />
                New analysis
              </button>
            )}
          </div>
          {page === "Overview" && (
            <>
              <div className="stats">
                <Stat
                  label="Saved recordings"
                  value={sessions.length}
                  note="Available in this team workspace"
                />
                <Stat
                  label="Completed analyses"
                  value={
                    sessions.filter((s) => s.status.state === "complete").length
                  }
                  note="Ready for footage review and export"
                />
                <Stat
                  label="Worker status"
                  value={sessions.some(active) ? "Processing" : "Available"}
                  note="CPU inference · one concurrent job"
                />
              </div>
              <div className="section-title">
                <h2>Recent analyses</h2>
                <button className="link" onClick={() => navigate("Sessions")}>
                  View all sessions <ArrowRight size={15} />
                </button>
              </div>
              {sessions.length ? (
                <SessionList
                  sessions={sessions.slice(0, 5)}
                  open={(s) => action(() => open(s))}
                  remove={remove}
                />
              ) : (
                <Empty
                  icon={<Film />}
                  title="Your first review starts here"
                  text="Upload a short campus recording, mark a counting zone and review sustained occupancy above your chosen capacity."
                  action={
                    <button
                      className="primary"
                      onClick={() => navigate("Analyze footage")}
                    >
                      Analyze footage <ArrowRight size={16} />
                    </button>
                  }
                />
              )}
              <div className="guide">
                <div>
                  <span>01</span>
                  <h3>Define the scene</h3>
                  <p>
                    Choose up to five named areas and set a head-count capacity
                    for each.
                  </p>
                </div>
                <div>
                  <span>02</span>
                  <h3>Review the evidence</h3>
                  <p>
                    Use synchronized playback and event markers to inspect
                    sustained buildup.
                  </p>
                </div>
                <div>
                  <span>03</span>
                  <h3>Share a clear record</h3>
                  <p>
                    Export sampled counts, analysis metadata and compressed
                    annotated footage.
                  </p>
                </div>
              </div>
              <p className="footnote">
                Decision support for recorded footage. Counts are estimates;
                capacity thresholds are configured by your team, not certified
                safety limits.
              </p>
            </>
          )}
          {page === "Sessions" && (
            <>
              <div className="filters">
                <label className="search">
                  <Search size={17} />
                  <input
                    aria-label="Search recordings"
                    placeholder="Search recordings…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                </label>
                <select
                  aria-label="Filter sessions by status"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                >
                  {[
                    "all",
                    "ready",
                    "running",
                    "complete",
                    "failed",
                    "cancelled",
                  ].map((v) => (
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
            </>
          )}
          {page === "Analyze footage" && (
            <>
              {!selected ? (
                <>
                  <div className="steps">
                    <b>1 / Upload recording</b>
                    <span>2 / Define zones</span>
                    <span>3 / Review findings</span>
                  </div>
                  <div className="upload-box">
                    <Upload size={32} />
                    <h2>Bring your event into focus</h2>
                    <p>Choose a recorded clip from a fixed campus camera.</p>
                    <label className="file-button">
                      {busy ? "Uploading…" : "Choose recording"}
                      <input
                        aria-label="Upload recording"
                        type="file"
                        accept=".mp4,.mov,.avi,.mkv"
                        disabled={busy}
                        onChange={(e) =>
                          e.target.files?.[0] && upload(e.target.files[0])
                        }
                      />
                    </label>
                    {uploadProgress !== null && (
                      <progress
                        aria-label="Upload progress"
                        value={uploadProgress}
                        max={1}
                      />
                    )}
                    <small>
                      MP4, MOV, AVI, MKV · up to 200 MB · up to 10 minutes · 4K
                      maximum
                    </small>
                  </div>
                  <div className="notice">
                    <ShieldCheck size={20} />
                    <div>
                      <b>Use footage you are authorized to review</b>
                      <p>
                        Media is restricted to your signed-in team. Automatic
                        retention: {retention} days. Avoid unnecessary personal
                        information in filenames.
                      </p>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="recording-bar">
                    <Film size={18} />
                    <b>{selected.file_name}</b>
                    <span>
                      {fmt(selected.duration)} · {mb(selected.bytes)}
                    </span>
                    <Badge state={selected.status.state} />
                    <button
                      className="link"
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
                </>
              )}
            </>
          )}
          {page === "Model evaluation" && evaluation && (
            <>
              <div className="notice">
                <Check size={20} />
                <div>
                  <b>Existing head detector retained</b>
                  <p>{evaluation.decision}</p>
                </div>
              </div>
              <div className="stats">
                <Stat
                  label="Counting MAE"
                  value={evaluation.baseline.mae.toFixed(2) + " heads"}
                  note="Mean absolute error per evaluation image"
                />
                <Stat
                  label="Counting RMSE"
                  value={evaluation.baseline.rmse.toFixed(2) + " heads"}
                  note="Penalizes larger counting errors"
                />
                <Stat
                  label="Evaluation images"
                  value={evaluation.baseline.images}
                  note="SCUT-HEAD Part B comparison set"
                />
              </div>
              <section className="panel">
                <h2>Reproducible model comparison</h2>
                <p>
                  Lower counting error is better. Detection precision and recall
                  use IoU 0.5.
                </p>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>MAE / heads</th>
                        <th>RMSE / heads</th>
                        <th>Precision</th>
                        <th>Recall</th>
                      </tr>
                    </thead>
                    <tbody>
                      {["baseline", "candidate"].map((k) => (
                        <tr key={k}>
                          <td>
                            {k === "baseline"
                              ? "Current detector"
                              : "Colab-trained candidate"}
                          </td>
                          <td>{evaluation[k].mae.toFixed(3)}</td>
                          <td>{evaluation[k].rmse.toFixed(3)}</td>
                          <td>{(evaluation[k].precision * 100).toFixed(1)}%</td>
                          <td>{(evaluation[k].recall * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
              <div className="two-col">
                <section className="panel">
                  <h2>Dataset & training</h2>
                  <p>{evaluation.dataset}</p>
                  <dl>
                    <dt>Training / tuning / comparison</dt>
                    <dd>1,440 / 456 / 494 images</dd>
                    <dt>Training hardware</dt>
                    <dd>{evaluation.hardware}</dd>
                    <dt>Completed epochs</dt>
                    <dd>{evaluation.completed_epochs}</dd>
                    <dt>Dataset license</dt>
                    <dd>{evaluation.dataset_license}</dd>
                  </dl>
                  <a
                    href={evaluation.dataset_source}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Official dataset documentation ↗
                  </a>
                </section>
                <section className="panel">
                  <h2>What these numbers cannot tell us</h2>
                  <p>{evaluation.limitation}</p>
                  <p>
                    This is not verified campus accuracy. Occlusion, camera
                    angle, blur and small heads can change results. Validate
                    representative campus clips with manually counted references
                    before operational use.
                  </p>
                  <p>
                    Heatmaps show detection frequency. Overlapping zones are
                    counted separately within each zone; the overall count uses
                    their union.
                  </p>
                </section>
              </div>
            </>
          )}
        </div>
        <footer>
          Campus Vision{" "}
          <span>Recorded-footage review · Research prototype</span>
        </footer>
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
                fill="#f4b94244"
                stroke="#f4b942"
                strokeWidth=".5"
              />
            )}
            {points.map((p, i) => (
              <circle
                key={i}
                cx={p[0] * 100}
                cy={p[1] * 100}
                r=".6"
                fill="#f4b942"
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
        <p className="hint">
          Click 3–12 corners clockwise to draw a convex area. Overlapping zones
          share detections; overall counts do not double-count them.
        </p>
        <details>
          <summary>Keyboard-accessible zone coordinates</summary>
          <p className="hint">
            Normalized coordinates: 0 is left/top; 1 is right/bottom. Edit each
            corner without drawing.
          </p>
          {current.polygon.map((p, i) => (
            <div className="coordinate" key={i}>
              <span>Corner {i + 1}</span>
              {[0, 1].map((axis) => (
                <label key={axis}>
                  {axis ? "Y" : "X"}
                  <input
                    aria-label={`Corner ${i + 1} ${axis ? "Y" : "X"}`}
                    type="number"
                    min="0"
                    max="1"
                    step=".01"
                    value={p[axis]}
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
            <Plus size={16} />
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
        <small className="saved">
          <Check size={13} /> Draft settings saved on this device
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
        <span className="quiet">
          <Check size={16} /> Completed · {r.settings.location}
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
          <div className="section-title">
            <h2>Footage review</h2>
            <button aria-pressed={heat} onClick={() => setHeat(!heat)}>
              {heat ? "Hide heatmap" : "Show activity heatmap"}
            </button>
          </div>
          <div className="video-stage">
            <video
              ref={video}
              controls
              playsInline
              preload="metadata"
              src={`/api/sessions/${session.id}/media/video`}
              onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
            />
            {heat && (
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
                        fill="#ffb020"
                        opacity={0.12 + (0.7 * v) / heatmax}
                      />
                    ) : null,
                  ),
                )}
              </svg>
            )}
          </div>
          {heat && (
            <p className="hint heat-legend">
              Low <span /> High · Relative detection frequency across{" "}
              {r.heatmap.samples} samples. Not physical density or unique
              people.
            </p>
          )}
          <div className="timeline-head">
            <h3>Occupancy timeline</h3>
            <label className="inline-label">
              Counting area
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
                <line key={y} x1="0" x2="800" y1={y} y2={y} stroke="#e4e9ed" />
              ))}
              {events.map((e, i) => (
                <rect
                  key={i}
                  x={(e.start_seconds / duration) * 800}
                  y="0"
                  width={Math.max(2, (e.duration_seconds / duration) * 800)}
                  height="150"
                  fill="#f5c779"
                  opacity=".25"
                />
              ))}
              {zone >= 0 && (
                <line
                  x1="0"
                  x2="800"
                  y1={145 - (r.zones[zone].capacity / max) * 135}
                  y2={145 - (r.zones[zone].capacity / max) * 135}
                  stroke="#b76c12"
                  strokeDasharray="5 5"
                />
              )}
              <polyline
                fill="none"
                stroke="#087f83"
                strokeWidth="2"
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
                stroke="#142b32"
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
          <div className="time-labels">
            <span>
              {fmt(time)} / {fmt(duration)}
            </span>
            <span>
              Teal: observed heads · Amber: buildup events
              {zone >= 0 ? " · Dashed: capacity" : ""}
            </span>
          </div>
        </section>
        <section className="panel event-panel">
          <div className="section-title">
            <h2>Review events</h2>
            <span className="count-label">{events.length}</span>
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
            <div className="small-empty">
              <Check size={22} />
              <h3>No sustained buildup detected</h3>
              <p>
                No sampled episode met this zone’s capacity and duration
                thresholds. This does not establish that the footage is safe.
              </p>
            </div>
          )}
        </section>
      </div>
      <details className="panel">
        <summary>Analysis record & settings used</summary>
        <p>
          These settings belong to this completed result. Your current draft
          settings are separate.
        </p>
        <div className="record-grid">
          <span>
            Sampling<b>{r.settings.analysis_fps} frames / second</b>
          </span>
          <span>
            Detector confidence<b>{r.settings.conf}</b>
          </span>
          <span>
            Analyzed frames<b>{r.processed_frames}</b>
          </span>
          <span>
            Output size
            <b>
              {mb(r.output_bytes)} / {mb(r.input_bytes)} input
            </b>
          </span>
        </div>
        {r.zones.map((z) => (
          <p key={z.name}>
            <b>{z.name}</b> · capacity {z.capacity} heads · observed peak{" "}
            {z.peak} heads
          </p>
        ))}
        <p className="hint">
          Counts may miss occluded heads or include false detections. Heatmap
          cells accumulate head-center observations; perspective is not
          calibrated. This is recorded-footage review, not a safety prediction.
        </p>
      </details>
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
