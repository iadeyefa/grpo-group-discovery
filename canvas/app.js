// Dynamic Application Logic for GRPO Group Discovery Canvas

window.currentSelectedDomainKey = 'social_values';

window.updateDomainMetrics = function(selectEl) {
  const data = window.GRPO_DATA;
  if (!data || !selectEl) return;

  const selectedKey = selectEl.value;
  window.currentSelectedDomainKey = selectedKey;

  const domainRun = data.runs.find(r => r.id === 'discovery_domain_topic_clustering');
  if (!domainRun || !domainRun.domain_breakdowns || !domainRun.domain_breakdowns[selectedKey]) return;

  const bd = domainRun.domain_breakdowns[selectedKey];

  // Update object metrics for sorting
  domainRun.lift_vs_pooled = bd.lift;
  domainRun.mean_jsd = bd.mean_jsd;
  domainRun.cohesion = bd.cohesion;
  domainRun.db_score = bd.db_score;
  domainRun.cluster_sizes = bd.sizes;
  domainRun.cluster_balance_desc = bd.desc;

  // Re-render and re-sort scorecard table
  if (typeof window.renderScorecardApp === 'function') {
    window.renderScorecardApp();
  }

  // Sync map dropdown and update map for selected topic domain
  const domMapKey = 'discovery_domain_' + selectedKey;
  const mapSelectEl = document.getElementById('map-model-select');
  if (mapSelectEl) {
    mapSelectEl.value = domMapKey;
  }
  if (typeof window.renderWorldMapApp === 'function') {
    window.renderWorldMapApp(domMapKey);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  const data = window.GRPO_DATA;
  if (!data) return;

  let sortField = 'lift_vs_pooled';
  let sortAscending = false;

  // Cache DOM elements
  const scorecardBody = document.getElementById('scorecard-body');
  const cardsGrid = document.getElementById('methods-cards-grid');

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
      
      let methodCellHTML = `<div class="method-name">${run.name}</div>`;
      let clusterBarsHTML = '';

      if (isOracle) {
        clusterBarsHTML = `<span style="font-size:0.75rem; color:var(--accent-cyan)">138 Singletons</span>`;
      } else if (run.domain_breakdowns) {
        const curKey = window.currentSelectedDomainKey || 'social_values';
        const bd = run.domain_breakdowns[curKey] || run.domain_breakdowns.social_values;

        const optionsHTML = Object.keys(run.domain_breakdowns).map(key => {
          const item = run.domain_breakdowns[key];
          const isSel = key === curKey ? 'selected' : '';
          return `<option value="${key}" ${isSel}>${item.name}</option>`;
        }).join('');

        const domainSelectHTML = `
          <div style="margin-top: 6px; display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 0.75rem; color: var(--accent-cyan); font-weight: 500;">Topic Domain:</span>
            <select class="domain-cluster-select" onchange="window.updateDomainMetrics(this)" style="background: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; cursor: pointer;">
              ${optionsHTML}
            </select>
          </div>
        `;

        methodCellHTML = `
          <div class="method-name">${run.name}</div>
          ${domainSelectHTML}
        `;

        const segments = bd.sizes.map(sz => {
          const pct = ((sz / totalEntities) * 100).toFixed(1);
          return `<div class="cluster-segment" style="width: ${pct}%;" title="Size: ${sz} (${pct}%)"></div>`;
        }).join('');

        clusterBarsHTML = `
          <div class="cluster-bar-container domain-bar-target">${segments}</div>
          <div class="domain-desc-target" style="font-size:0.75rem; color:var(--text-dim); margin-top:4px;">${bd.desc}</div>
        `;
      } else {
        const segments = run.cluster_sizes.slice(0, 5).map(sz => {
          const pct = ((sz / totalEntities) * 100).toFixed(1);
          return `<div class="cluster-segment" style="width: ${pct}%;" title="Size: ${sz} (${pct}%)"></div>`;
        }).join('');
        clusterBarsHTML = `
          <div class="cluster-bar-container">${segments}</div>
          <div style="font-size:0.75rem; color:var(--text-dim); margin-top:4px;">${run.cluster_balance_desc}</div>
        `;
      }

      return `
        <tr data-run-id="${run.id}">
          <td>
            ${methodCellHTML}
          </td>
          <td class="cell-lift">
            <span class="metric-highlight ${liftClass}">${liftFormatted}</span>
          </td>
          <td class="cell-jsd">${run.mean_jsd.toFixed(2)}</td>
          <td class="cell-cohesion">${run.cohesion.toFixed(3)}</td>
          <td class="cell-db">${run.db_score.toFixed(2)}</td>
          <td>
            ${clusterBarsHTML}
          </td>
        </tr>
      `;
    }).join('');
  }

  window.renderScorecardApp = renderScorecard;

  // Render Method Deep Dive Cards
  function renderMethodCards() {
    cardsGrid.innerHTML = data.method_deep_dives.map(card => {
      const strengthsList = card.key_strengths.map(s => `<li><span style="color:#6366f1; margin-right:8px; font-size:1.1rem;">&bull;</span>${s}</li>`).join('');
      const weaknessesList = card.key_weaknesses.map(w => `<li><span style="color:#6366f1; margin-right:8px; font-size:1.1rem;">&bull;</span>${w}</li>`).join('');

      return `
        <div class="method-card" id="card-${card.id}">
          <div class="card-header">
            <h4>${card.title}</h4>
          </div>

          <strong style="font-size: 0.8rem; color: var(--text-muted);">Key Strengths:</strong>
          <ul class="bullets-list">${strengthsList}</ul>

          <strong style="font-size: 0.8rem; color: var(--text-muted);">Key Failures / Weaknesses:</strong>
          <ul class="bullets-list">${weaknessesList}</ul>
        </div>
      `;
    }).join('');
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
    const hoverText = mapData.map(d => d.hover_text);

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

  window.renderWorldMapApp = renderWorldMap;

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
});
