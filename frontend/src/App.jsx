import {
  Activity,
  CheckCircle2,
  Database,
  ListTree,
  Play,
  RefreshCw,
  Search,
  SlidersHorizontal,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

const initialConfig = {
  data_dir: "data/uf20-91",
  timesteps: 100000,
  seed: 0,
  max_steps: 40,
  n_steps: 2048,
  batch_size: 128,
  gamma: 1,
  n_envs: 8,
  lr_decay: true,
  model_out: "",
  rewards: {
    invalid_action_penalty: -1,
    solved_bonus: 10,
    failed_penalty: -10,
    falsified_clause_penalty: -0.5,
    unit_clause_bonus: 2,
  },
};

function App() {
  const [config, setConfig] = useState(initialConfig);
  const [environment, setEnvironment] = useState(null);
  const [models, setModels] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [activeJob, setActiveJob] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [episodes, setEpisodes] = useState(100);
  const [message, setMessage] = useState("");

  const progressPercent = useMemo(() => {
    if (!activeJob?.progress) return 0;
    return Math.min(
      100,
      Math.round(
        (activeJob.progress.current_timesteps / activeJob.progress.total_timesteps) * 100,
      ),
    );
  }, [activeJob]);

  useEffect(() => {
    refreshEnvironment();
    refreshModels();
    refreshJobs();
  }, []);

  useEffect(() => {
    if (!activeJob || activeJob.status !== "running") return undefined;
    const timer = window.setInterval(() => refreshJob(activeJob.id), 1500);
    return () => window.clearInterval(timer);
  }, [activeJob]);

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail ?? `HTTP ${response.status}`);
    }
    return response.json();
  }

  async function refreshEnvironment() {
    try {
      const params = new URLSearchParams({ data_dir: config.data_dir });
      setEnvironment(await request(`/api/environment?${params}`));
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function refreshModels() {
    try {
      const data = await request("/api/models");
      setModels(data.models);
      if (data.models.length > 0) {
        setSelectedModel((current) => current || data.models[0].path);
      }
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function refreshJobs() {
    try {
      const data = await request("/api/train");
      setJobs(data);
      const running = data.find((job) => job.status === "running");
      if (running) setActiveJob(running);
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function startTraining() {
    setMessage("");
    setEvaluation(null);
    try {
      const payload = {
        ...config,
        model_out: config.model_out.trim() ? config.model_out.trim() : null,
      };
      const job = await request("/api/train", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setActiveJob(job);
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function refreshJob(jobId) {
    try {
      const job = await request(`/api/train/${jobId}`);
      setActiveJob(job);
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      if (job.status === "completed") {
        await refreshModels();
        setSelectedModel(job.model_path);
      }
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function evaluateModel() {
    setMessage("");
    if (!selectedModel) {
      setMessage("Choose a model checkpoint before evaluating.");
      return;
    }
    try {
      const result = await request("/api/evaluate", {
        method: "POST",
        body: JSON.stringify({
          data_dir: config.data_dir,
          model: selectedModel,
          episodes,
          max_steps: config.max_steps,
          seed: config.seed,
          rewards: config.rewards,
        }),
      });
      setEvaluation(result);
    } catch (error) {
      setMessage(error.message);
    }
  }

  function updateField(name, value) {
    setConfig((current) => ({ ...current, [name]: value }));
  }

  function updateReward(name, value) {
    setConfig((current) => ({
      ...current,
      rewards: { ...current.rewards, [name]: value },
    }));
  }

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">uf20-91 PPO Lab</p>
          <h1>saharSAT</h1>
        </div>
        <button className="icon-button" type="button" onClick={() => {
          refreshEnvironment();
          refreshModels();
          refreshJobs();
        }}>
          <RefreshCw size={18} />
          Refresh
        </button>
      </section>

      {message && <div className="notice">{message}</div>}

      <section className="workspace-grid">
        <div className="panel config-panel">
          <div className="panel-heading">
            <SlidersHorizontal size={18} />
            <h2>Training Config</h2>
          </div>

          <label>
            Dataset
            <input
              value={config.data_dir}
              onChange={(event) => updateField("data_dir", event.target.value)}
              onBlur={refreshEnvironment}
            />
          </label>

          <div className="form-grid">
            <NumberField label="Timesteps" value={config.timesteps} min={1} onChange={(value) => updateField("timesteps", value)} />
            <NumberField label="Seed" value={config.seed} onChange={(value) => updateField("seed", value)} />
            <NumberField label="Max steps" value={config.max_steps} min={1} onChange={(value) => updateField("max_steps", value)} />
            <NumberField label="Rollout steps" value={config.n_steps} min={1} onChange={(value) => updateField("n_steps", value)} />
            <NumberField label="Batch size" value={config.batch_size} min={1} onChange={(value) => updateField("batch_size", value)} />
            <NumberField label="Gamma" value={config.gamma} step={0.01} min={0.01} max={1} onChange={(value) => updateField("gamma", value)} />
            <NumberField label="Parallel envs" value={config.n_envs} min={1} onChange={(value) => updateField("n_envs", value)} />
          </div>

          <div className="reward-grid">
            <NumberField label="Invalid action" value={config.rewards.invalid_action_penalty} step={0.5} onChange={(value) => updateReward("invalid_action_penalty", value)} />
            <NumberField label="Solved bonus" value={config.rewards.solved_bonus} step={5} onChange={(value) => updateReward("solved_bonus", value)} />
            <NumberField label="Failed penalty" value={config.rewards.failed_penalty} step={5} onChange={(value) => updateReward("failed_penalty", value)} />
            <NumberField label="Falsified clause" value={config.rewards.falsified_clause_penalty} step={0.1} onChange={(value) => updateReward("falsified_clause_penalty", value)} />
            <NumberField label="Unit clause" value={config.rewards.unit_clause_bonus} step={0.25} onChange={(value) => updateReward("unit_clause_bonus", value)} />
          </div>

          <label className="toggle-row">
            <input
              type="checkbox"
              checked={config.lr_decay}
              onChange={(event) => updateField("lr_decay", event.target.checked)}
            />
            Linear learning-rate decay
          </label>

          <label>
            Model output
            <input
              placeholder="models/ppo_uf20_91_custom.zip"
              value={config.model_out}
              onChange={(event) => updateField("model_out", event.target.value)}
            />
          </label>

          <button
            className="primary-button"
            type="button"
            onClick={startTraining}
            disabled={activeJob?.status === "running"}
          >
            <Play size={18} />
            Start Training
          </button>
        </div>

        <div className="panel status-panel">
          <div className="panel-heading">
            <Activity size={18} />
            <h2>Training</h2>
          </div>

          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
          </div>

          <div className="metric-grid">
            <Metric label="Status" value={activeJob?.status ?? "idle"} />
            <Metric label="Progress" value={`${progressPercent}%`} />
            <Metric label="Timesteps" value={activeJob?.progress ? `${activeJob.progress.current_timesteps}/${activeJob.progress.total_timesteps}` : "0"} />
            <Metric label="Mean episode reward" value={formatMaybe(activeJob?.progress?.last_mean_reward)} />
            <Metric label="Mean episode length" value={formatMaybe(activeJob?.progress?.last_mean_episode_length)} />
            <Metric label="Updated" value={activeJob?.progress?.updated_at ?? "-"} />
            <Metric label="Model" value={activeJob?.model_path ?? "-"} wide />
            {activeJob?.error && <Metric label="Error" value={activeJob.error} wide />}
          </div>
        </div>

        <div className="panel jobs-panel">
          <div className="panel-heading">
            <ListTree size={18} />
            <h2>Jobs</h2>
          </div>

          <div className="job-list">
            {jobs.length === 0 && <p className="empty-state">No training jobs in this API session.</p>}
            {jobs.map((job) => (
              <button
                className={activeJob?.id === job.id ? "job-row job-row-active" : "job-row"}
                key={job.id}
                type="button"
                onClick={() => setActiveJob(job)}
              >
                <StatusIcon status={job.status} />
                <span>
                  <strong>{job.status}</strong>
                  <small>{job.config.timesteps.toLocaleString()} steps · {job.config.n_envs} envs</small>
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="panel env-panel">
          <div className="panel-heading">
            <Database size={18} />
            <h2>Environment</h2>
          </div>
          <div className="metric-grid">
            <Metric label="Instances" value={environment?.instances ?? "-"} />
            <Metric label="Variables" value={environment?.num_vars ?? "-"} />
            <Metric label="Clauses" value={environment?.num_clauses ?? "-"} />
            <Metric label="Actions" value={environment?.action_space ?? "-"} />
            <Metric label="Observation" value={environment?.observation_shape?.join(" x ") ?? "-"} />
            <Metric label="Default steps" value={environment?.default_max_steps ?? "-"} />
          </div>
        </div>

        <div className="panel eval-panel">
          <div className="panel-heading">
            <Search size={18} />
            <h2>Evaluation</h2>
          </div>

          <label>
            Model
            <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              {!selectedModel && <option value="">Choose checkpoint</option>}
              {models.map((model) => (
                <option key={model.path} value={model.path}>
                  {model.name} ({formatBytes(model.size_bytes)})
                </option>
              ))}
            </select>
          </label>

          <NumberField label="Episodes" value={episodes} min={1} onChange={setEpisodes} />

          <button className="secondary-button" type="button" onClick={evaluateModel}>
            <Search size={18} />
            Evaluate
          </button>

          <div className="metric-grid">
            <Metric label="Solved" value={evaluation ? `${evaluation.solved}/${evaluation.episodes}` : "-"} />
            <Metric label="Solve rate" value={evaluation ? `${Math.round(evaluation.solve_rate * 100)}%` : "-"} />
            <Metric label="Mean clauses" value={evaluation ? evaluation.satisfied_clauses.mean.toFixed(2) : "-"} />
            <Metric label="Steps" value={evaluation ? evaluation.steps.mean.toFixed(2) : "-"} />
            <Metric label="Clause range" value={evaluation ? `${evaluation.satisfied_clauses.min}-${evaluation.satisfied_clauses.max}` : "-"} />
            <Metric label="Step range" value={evaluation ? `${evaluation.steps.min}-${evaluation.steps.max}` : "-"} />
          </div>
        </div>
      </section>
    </main>
  );
}

function NumberField({ label, value, onChange, min, max, step = 1 }) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function Metric({ label, value, wide = false }) {
  return (
    <div className={wide ? "metric metric-wide" : "metric"}>
      <span>{label}</span>
      <strong title={String(value)}>{value}</strong>
    </div>
  );
}

function StatusIcon({ status }) {
  if (status === "completed") return <CheckCircle2 size={18} className="status-completed" />;
  if (status === "failed") return <XCircle size={18} className="status-failed" />;
  return <Activity size={18} className="status-running" />;
}

function formatMaybe(value) {
  if (value === null || value === undefined) return "-";
  return Number(value).toFixed(2);
}

function formatBytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export default App;
