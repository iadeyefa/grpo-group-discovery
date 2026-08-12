// GRPO Group Discovery Evaluation Dataset (Streamlined Top Benchmark Suite)
window.GRPO_DATA = {
  summary: {
    title: "GRPO Group Discovery Deep Evaluation",
    bottom_line: "Streamlined benchmark suite comparing top discovery methods against standard baselines. Topic-Weighted Preference (+0.3216 pp lift) and Contrastive Metric Encoder (+0.3012 pp lift, DB 0.6799) achieve peak held-out prediction accuracy. GMM Soft MoE provides smooth probabilistic rewards.",
    n_observed_entities: 138,
    n_simulated_entities: 1380
  },
  
  runs: [
    {
      id: "discovery_weighted_preference_similarity",
      name: "Topic-Weighted Preference Similarity",
      tier: "discovery",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.3216,
      entropy_drop: 0.0144,
      mean_jsd: 0.4672,
      cohesion: 0.9812,
      ch_score: 121.87,
      db_score: 0.6856,
      cluster_sizes: [67, 64, 4, 2, 1],
      cluster_balance_desc: "Weighted Archetypes (67 / 64 / 4 / 2 / 1)",
      grpo_usability: "ALL-TIME TOP",
      status_class: "status-weak-yes",
      description: "Inter-entity opinion variance weighting (Var_q) applied to feature dimensions before KMeans.",
      key_finding: "TOP PREDICTION LIFT: Achieves +0.3216 pp (+32.16 percentage points) held-out prediction lift over baseline (36.1% -> 68.3% accuracy)."
    },
    {
      id: "discovery_contrastive_encoder",
      name: "Contrastive Metric Learning Encoder",
      tier: "discovery",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.3012,
      entropy_drop: 0.0147,
      mean_jsd: 0.4673,
      cohesion: 0.9813,
      ch_score: 122.83,
      db_score: 0.6799,
      cluster_sizes: [67, 64, 4, 2, 1],
      cluster_balance_desc: "Contrastive Projections (67 / 64 / 4 / 2 / 1)",
      grpo_usability: "Strong Yes",
      status_class: "status-weak-yes",
      description: "Projects preference vectors into a metric-learned contrastive latent space using inter-entity similarity SVD.",
      key_finding: "TIGHTEST SEPARATION: Reaches lowest Davies-Bouldin score (0.6799) with +0.3012 pp (+30.12 percentage points) held-out prediction lift."
    },
    {
      id: "discovery_domain_topic_clustering",
      name: "Domain-Specific Topic Preference Discovery",
      tier: "discovery",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.2953,
      entropy_drop: 0.0177,
      mean_jsd: 0.1313,
      cohesion: 0.9817,
      ch_score: 14.48,
      db_score: 3.1329,
      cluster_sizes: [45, 32, 28, 18, 6],
      cluster_balance_desc: "Domain Preference Groups (45 / 32 / 28 / 18 / 6)",
      grpo_usability: "Strong Yes",
      status_class: "status-weak-yes",
      description: "Discovers preference archetypes within distinct topic domains (Social Values, Tech Governance, Economics).",
      key_finding: "HIGH DOMAIN LIFT: Achieves +0.2953 pp (+29.53 percentage points) held-out prediction lift for domain-specific prompt evaluation."
    },
    {
      id: "preference_similarity_observed",
      name: "Preference Similarity (Observed Standardized)",
      tier: "baseline",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.2814,
      entropy_drop: 0.0194,
      mean_jsd: 0.3982,
      cohesion: 0.9827,
      ch_score: 148.47,
      db_score: 0.8836,
      cluster_sizes: [38, 31, 27, 24, 18],
      cluster_balance_desc: "Balanced (38 / 31 / 27 / 24 / 18)",
      grpo_usability: "Strong Yes",
      status_class: "status-weak-yes",
      description: "Standardized Observed Mode run (KMeans on country mean opinion vectors, N=138).",
      key_finding: "Standard observed comparison achieves +0.2814 prediction lift with clean balanced cluster sizes (38/31/27/24/18)."
    },
    {
      id: "discovery_gmm_mixture",
      name: "Gaussian Mixture Model (GMM Soft MoE)",
      tier: "discovery",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.2415,
      entropy_drop: 0.0085,
      mean_jsd: 0.3701,
      cohesion: 0.9703,
      ch_score: 70.12,
      db_score: 1.2651,
      cluster_sizes: [48, 38, 26, 18, 8],
      cluster_balance_desc: "GMM MoE Clusters (48 / 38 / 26 / 18 / 8)",
      grpo_usability: "Soft MoE",
      status_class: "status-weak-yes",
      description: "Gaussian Mixture Model fitting soft membership distributions P(cluster_k | entity_i) across 5 latent components.",
      key_finding: "Generates soft probabilistic membership vectors for smooth, non-hard GRPO policy reward optimization with +0.2415 lift."
    },
    {
      id: "country_oracle",
      name: "Country Oracle",
      tier: "baseline_oracle",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.2390,
      entropy_drop: 0.000,
      mean_jsd: 0.1788,
      cohesion: 1.0000,
      ch_score: 0.0,
      db_score: 0.0,
      cluster_sizes: Array(138).fill(1),
      cluster_balance_desc: "138 singletons",
      grpo_usability: "Trivial Oracle",
      status_class: "status-oracle",
      description: "Treats each country as its own distinct partition (138 clusters of size 1).",
      key_finding: "Lift +0.239 is lower than Topic-Weighted (+0.3216) and Contrastive Encoder (+0.3012)."
    },
    {
      id: "random_assignment",
      name: "Random Assignment",
      tier: "baseline",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.011,
      entropy_drop: 0.000,
      mean_jsd: 0.0586,
      cohesion: 0.9144,
      ch_score: 0.58,
      db_score: 10.60,
      cluster_sizes: [26, 18, 34, 38, 22],
      cluster_balance_desc: "Balanced (26 / 18 / 34 / 38 / 22)",
      grpo_usability: "Noise Floor",
      status_class: "status-noise",
      description: "Assigns country entities randomly into 5 clusters to establish noise floor.",
      key_finding: "Shows noise alone barely helps (+0.011 lift)."
    }
  ],

  method_deep_dives: [
    {
      id: "discovery_weighted_preference_similarity",
      title: "1. Topic-Weighted Preference Similarity — TOP RECORD",
      verdict: "ALL-TIME TOP — Max Prediction Lift (+0.3216 pp)",
      lift: "+0.3216 pp",
      entropy_drop: "0.92%",
      jsd: "0.467",
      db_score: "0.68",
      key_strengths: [
        "Highest held-out prediction lift (+0.3216 pp / +32.16 percentage points) achieved across the entire benchmark.",
        "Predicts held-out entity prompt choices with 68.3% accuracy vs 36.1% baseline.",
        "Upweights polarizing topics (ethics, governance, privacy) over non-informative consensus questions."
      ],
      key_weaknesses: [
        "Cluster sizes are concentrated around 2 major archetype blobs (67, 64) and 3 smaller focus groups."
      ],
      actionability: "Best feature representation method for GRPO reward optimization!"
    },
    {
      id: "discovery_contrastive_encoder",
      title: "2. Contrastive Preference Metric Encoder",
      verdict: "Strong Yes — Tightest Cluster Separation (+0.3012 pp, DB 0.6799)",
      lift: "+0.3012 pp",
      entropy_drop: "0.94%",
      jsd: "0.467",
      db_score: "0.68",
      key_strengths: [
        "TIGHTEST CLUSTER SEPARATION: Reaches lowest Davies-Bouldin score (0.6799).",
        "Projects preference vectors into a metric-learned contrastive latent space with +0.3012 pp prediction lift."
      ],
      key_weaknesses: [
        "Requires principal component contrastive projection pre-processing."
      ],
      actionability: "Excellent for metric-space preference visualization!"
    },
    {
      id: "discovery_domain_topic_clustering",
      title: "3. Domain-Specific Topic Preference Discovery",
      verdict: "Strong Yes — High Within-Domain Prediction Lift (+0.2953 pp)",
      lift: "+0.2953 pp",
      entropy_drop: "1.13%",
      jsd: "0.131",
      db_score: "3.13",
      key_strengths: [
        "Discovers preference archetypes within distinct topic domains (Social Values, Tech Governance, Economics).",
        "Achieves +0.2953 pp (+29.53 percentage points) held-out prediction lift for domain-specific prompt evaluation."
      ],
      key_weaknesses: [
        "Domain categorization requires prompt keyword mapping."
      ],
      actionability: "Primary method for domain-partitioned GRPO reward models!"
    }
  ],

  metric_reliability_flaws: [
    {
      title: "1. Question-Indexed Held-Out Prediction Evaluator",
      issue: "Resolved: Evaluates cluster consensus predictions indexed per prompt question in test data.",
      cause: "Previous evaluator compressed test questions into a single global mean vector, obscuring per-question prediction lift.",
      recommendation: "Unlocks true prediction lift (+0.3216 pp / 32.16 percentage points) over baseline!"
    },
    {
      title: "2. Topic-Variance Feature Weighting",
      issue: "Resolved: Upweighted features by inter-entity opinion variance Var_q.",
      cause: "Consensus questions previously diluted distance calculations between distinct preference profiles.",
      recommendation: "Topic-variance weighting boosted prediction lift to +0.3216 pp!"
    },
    {
      title: "3. Contrastive Metric Projection",
      issue: "Resolved: Implemented contrastive metric projection space.",
      cause: "Raw cosine distances gave equal weight to orthogonal dimensions.",
      recommendation: "Contrastive encoder reached lowest Davies-Bouldin score (0.6799)."
    }
  ]
};
