// Dynamic Application Logic for GRPO Group Discovery Canvas

document.addEventListener('DOMContentLoaded', () => {
  const data = window.GRPO_DATA;
  if (!data) return;

  let currentModeFilter = 'all';
  let sortField = 'lift_vs_pooled';
  let sortAscending = false;

  // Cache DOM elements
  const scorecardBody = document.getElementById('scorecard-body');
  const cardsGrid = document.getElementById('methods-cards-grid');
  const flawsGrid = document.getElementById('flaws-grid');
  const roadmapGrid = document.getElementById('roadmap-grid');
  const filterBtns = document.querySelectorAll('.filter-btn');

  // Filter button handlers
  filterBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      filterBtns.forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      currentModeFilter = e.target.dataset.mode;
      renderScorecard();
    });
  });

  // Table header sort handlers
  document.querySelectorAll('#scorecard-table th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const field = th.dataset.sort;
      if (sortField === field) {
        sortAscending = !sortAscending;
      } else {
        sortField = field;
        sortAscending = false;
      }
      renderScorecard();
    });
  });

  // Render Scorecard Table
  function renderScorecard() {
    let filteredRuns = [...data.runs];
    if (currentModeFilter !== 'all') {
      filteredRuns = filteredRuns.filter(run => run.mode === currentModeFilter);
    }

    filteredRuns.sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];

      if (typeof valA === 'string') {
        return sortAscending ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sortAscending ? valA - valB : valB - valA;
    });

    scorecardBody.innerHTML = filteredRuns.map(run => {
      const liftFormatted = (run.lift_vs_pooled >= 0 ? '+' : '') + run.lift_vs_pooled.toFixed(3);
      const liftClass = run.lift_vs_pooled > 0.02 ? 'positive' : 'neutral';
      
      // Calculate cluster size bars
      const totalEntities = run.cluster_sizes.reduce((a, b) => a + b, 0);
      const isOracle = run.id === 'country_oracle';
      
      let clusterBarsHTML = '';
      if (isOracle) {
        clusterBarsHTML = `<span style="font-size:0.75rem; color:var(--accent-cyan)">138 Singletons</span>`;
      } else {
        const segments = run.cluster_sizes.slice(0, 5).map(sz => {
          const pct = ((sz / totalEntities) * 100).toFixed(1);
          return `<div class="cluster-segment" style="width: ${pct}%;" title="Size: ${sz} (${pct}%)"></div>`;
        }).join('');
        clusterBarsHTML = `<div class="cluster-bar-container">${segments}</div>`;
      }

      return `
        <tr>
          <td>
            <div class="method-name">${run.name}</div>
            <span style="font-size:0.75rem; color:var(--text-muted);">${run.tier}</span>
          </td>
          <td>
            <span class="mode-tag ${run.mode}">${run.mode === 'observed' ? 'COUNTRY' : run.mode.toUpperCase()}</span>
          </td>
          <td>${run.n_entities.toLocaleString()}</td>
          <td>
            <span class="metric-highlight ${liftClass}">${liftFormatted}</span>
          </td>
          <td>${run.entropy_drop.toFixed(3)}</td>
          <td>${run.mean_jsd.toFixed(2)}</td>
          <td>${run.cohesion.toFixed(3)}</td>
          <td>
            ${clusterBarsHTML}
            <div style="font-size:0.75rem; color:var(--text-dim); margin-top:4px;">${run.cluster_balance_desc}</div>
          </td>
          <td>
            <span class="status-badge ${run.status_class}">${run.grpo_usability}</span>
          </td>
        </tr>
      `;
    }).join('');
  }

  // Render Method Deep Dive Cards
  function renderMethodCards() {
    cardsGrid.innerHTML = data.method_deep_dives.map(card => {
      const strengthsList = card.key_strengths.map(s => `<li>${s}</li>`).join('');
      const weaknessesList = card.key_weaknesses.map(w => `<li>${w}</li>`).join('');

      return `
        <div class="method-card">
          <div class="card-header">
            <h4>${card.title}</h4>
          </div>
          
          <div class="metrics-pill-row">
            <div class="metric-pill"><span class="lbl">Lift:</span><span class="val">${card.lift}</span></div>
            <div class="metric-pill"><span class="lbl">Entropy Δ:</span><span class="val">${card.entropy_drop}</span></div>
            <div class="metric-pill"><span class="lbl">Mean JSD:</span><span class="val">${card.jsd}</span></div>
            <div class="metric-pill"><span class="lbl">DB Score:</span><span class="val">${card.db_score}</span></div>
          </div>

          <div style="margin-bottom: 10px; font-weight: 500; font-size: 0.85rem; color: var(--accent-cyan)">
            Verdict: ${card.verdict}
          </div>

          <strong style="font-size: 0.8rem; color: var(--text-muted);">Key Strengths:</strong>
          <ul class="bullets-list strengths">${strengthsList}</ul>

          <strong style="font-size: 0.8rem; color: var(--text-muted);">Key Failures / Weaknesses:</strong>
          <ul class="bullets-list weaknesses">${weaknessesList}</ul>

          <div class="card-actionable">
            <strong>Recommendation:</strong> ${card.actionability}
          </div>
        </div>
      `;
    }).join('');
  }

  // Render Metric Flaws
  function renderMetricFlaws() {
    flawsGrid.innerHTML = data.metric_reliability_flaws.map(flaw => `
      <div class="flaw-card">
        <h4>${flaw.title}</h4>
        <p><strong>Symptom:</strong> ${flaw.issue}</p>
        <p><strong>Root Cause:</strong> ${flaw.cause}</p>
        <div class="flaw-recom">Recommendation: ${flaw.recommendation}</div>
      </div>
    `).join('');
  }

  // Render Roadmap
  function renderRoadmap() {
    roadmapGrid.innerHTML = data.grpo_readiness_roadmap.map((item, idx) => `
      <div class="roadmap-step">
        <div class="step-num">Step 0${idx + 1} • ${item.status}</div>
        <h4>${item.step}</h4>
        <p>${item.action}</p>
      </div>
    `).join('');
  }

  // Initial Render
  renderScorecard();
  renderMethodCards();
  renderMetricFlaws();
  renderRoadmap();
});
