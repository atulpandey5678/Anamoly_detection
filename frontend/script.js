const CLUSTER_COLORS = [
  "#6366f1", "#ec4899", "#f59e0b",
  "#10b981", "#3b82f6", "#a78bfa", "#f43f5e",
];

async function runPipeline() {
  const btn     = document.getElementById("runBtn");
  const btnText = document.getElementById("btnText");
  const loader  = document.getElementById("btnLoader");
  const results = document.getElementById("results");
  const errBanner = document.getElementById("errorBanner");
  const errMsg    = document.getElementById("errorMsg");

  btn.disabled = true;
  btnText.textContent = "Running…";
  loader.classList.remove("hidden");
  errBanner.classList.add("hidden");
  results.classList.add("hidden");

  try {
    const res  = await fetch("/api/run", { method: "POST" });
    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || `Server error ${res.status}`);
    }

    populateDashboard(data);
    results.classList.remove("hidden");
    results.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    errMsg.textContent = err.message;
    errBanner.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btnText.textContent = "▶ Run Analysis";
    loader.classList.add("hidden");
  }
}

function populateDashboard(data) {
  // Stats
  animateCount("stat-total",    data.total_customers);
  animateCount("stat-anomalies", data.total_anomalies);
  document.getElementById("stat-clusters").textContent  = data.n_clusters;
  document.getElementById("stat-fraction").textContent  = (data.anomaly_fraction * 100).toFixed(0) + "%";

  // Plot
  if (data.plot_base64) {
    document.getElementById("plotImg").src = "data:image/png;base64," + data.plot_base64;
  }

  // Cluster bars
  renderClusterBars(data.cluster_counts, data.total_customers);

  // Table
  renderAnomalyTable(data.anomalies);
}

function animateCount(id, target) {
  const el = document.getElementById(id);
  const duration = 700;
  const start = performance.now();
  const from = 0;

  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(from + (target - from) * eased);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

function renderClusterBars(clusterCounts, total) {
  const container = document.getElementById("clusterBars");
  container.innerHTML = "";

  const sorted = Object.entries(clusterCounts).sort((a, b) => Number(a[0]) - Number(b[0]));
  const max = Math.max(...sorted.map(([, v]) => v));

  sorted.forEach(([cluster, count], i) => {
    const pct = total > 0 ? (count / total) * 100 : 0;
    const barWidth = max > 0 ? (count / max) * 100 : 0;
    const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];

    const row = document.createElement("div");
    row.className = "cluster-bar-row";
    row.innerHTML = `
      <div class="cluster-bar-label">
        <span class="cluster-dot" style="background:${color}"></span>Cluster ${cluster}
      </div>
      <div class="cluster-bar-track">
        <div class="cluster-bar-fill" style="width:0%; background:${color}" data-target="${barWidth}"></div>
      </div>
      <div class="cluster-bar-count">${count} <span style="color:var(--muted);font-weight:400">(${pct.toFixed(1)}%)</span></div>
    `;
    container.appendChild(row);
  });

  // Animate bars after paint
  requestAnimationFrame(() => {
    container.querySelectorAll(".cluster-bar-fill").forEach(el => {
      el.style.width = el.dataset.target + "%";
    });
  });
}

function renderAnomalyTable(anomalies) {
  const tbody = document.getElementById("anomalyBody");
  tbody.innerHTML = "";

  if (!anomalies || anomalies.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:32px">No anomalies found.</td></tr>`;
    return;
  }

  anomalies.forEach((row, i) => {
    const color = CLUSTER_COLORS[row.Cluster % CLUSTER_COLORS.length];
    const tr = document.createElement("tr");
    tr.style.animationDelay = `${i * 30}ms`;
    tr.innerHTML = `
      <td><strong>#${row.CustomerID}</strong></td>
      <td>$${row.Income}k</td>
      <td>${row["Spending Score"]}</td>
      <td>
        <span style="font-family:monospace;color:var(--accent2)">${row.Distance.toFixed(3)}</span>
      </td>
      <td>
        <span class="cluster-dot" style="background:${color}"></span>${row.Cluster}
      </td>
      <td><span class="badge-anomaly">🚨 Anomaly</span></td>
    `;
    tbody.appendChild(tr);
  });
}
