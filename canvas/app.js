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
            <span class="mode-tag ${run.mode}">${run.mode}</span>
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

  // Render Interactive World Map
  function renderWorldMap(selectedModelId = 'discovery_weighted_preference_similarity') {
    const allModelsData = window.WORLD_MAP_ALL_MODELS;
    const mapContainer = document.getElementById('world-map-container');
    if (!allModelsData || !mapContainer || typeof Plotly === 'undefined') return;

    const modelObj = allModelsData[selectedModelId];
    if (!modelObj || !modelObj.records) return;

    const mapData = modelObj.records;
    const locations = mapData.map(d => d.iso_alpha);
    const z = mapData.map(d => d.cluster_id);
    const hoverText = mapData.map(d => `<b>${d.source_group}</b><br>Cluster ${d.cluster_id}`);

    const colorScale = [
      [0.0, '#10b981'],  // Cluster 0 (Emerald)
      [0.25, '#8b5cf6'], // Cluster 1 (Purple)
      [0.50, '#3b82f6'], // Cluster 2 (Blue)
      [0.75, '#f59e0b'], // Cluster 3 (Amber)
      [1.0, '#ef4444']   // Cluster 4 (Red)
    ];

    const data = [{
      type: 'choropleth',
      locations: locations,
      z: z,
      text: hoverText,
      hoverinfo: 'text',
      colorscale: colorScale,
      showscale: false,
      marker: {
        line: {
          color: '#1e293b',
          width: 0.5
        }
      }
    }];

    const layout = {
      margin: { r: 0, t: 10, l: 0, b: 10 },
      paper_bgcolor: '#0f172a',
      plot_bgcolor: '#0f172a',
      geo: {
        bgcolor: '#0f172a',
        lakecolor: '#1e293b',
        showlakes: true,
        showcoastlines: true,
        coastlinecolor: '#334155',
        showframe: false,
        showcountries: true,
        countrycolor: '#334155',
        projection: { type: 'natural earth' }
      }
    };

    Plotly.newPlot('world-map-container', data, layout, { responsive: true, displayModeBar: false });
  }

  // Model Select Dropdown Handler
  const mapSelectEl = document.getElementById('map-model-select');
  if (mapSelectEl) {
    mapSelectEl.addEventListener('change', (e) => {
      renderWorldMap(e.target.value);
    });
  }

  // Initial Render
  renderScorecard();
  renderWorldMap('discovery_weighted_preference_similarity');
  renderMethodCards();
  renderMetricFlaws();
  renderRoadmap();
});
