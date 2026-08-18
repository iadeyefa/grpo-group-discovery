// GRPO Group Discovery Evaluation Dataset
window.GRPO_DATA = {
  summary: {
    title: "GRPO Group Discovery Evaluation",
    bottom_line: "We measure how accurately discovered belief groups predict a country's stances on held-out (unseen) survey prompts. We hide 20% of questions during clustering, calculate each group's average stance on the remaining 80% training questions, and then test how accurately a group's average consensus predicts member countries' answers on the unseen test questions (+0.3216 max lift, 36.1% to 68.3% accuracy).",
    n_observed_entities: 138,
    n_individual_entities: 1380
  },
  
  runs: [
    {
      id: "discovery_simulated_contrastive_encoder",
      name: "Individual Contrastive Metric Encoder",
      tier: "discovery",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.3216,
      mean_jsd: 0.4672,
      cohesion: 0.9820,
      ch_score: 1259.90,
      db_score: 0.6856,
      cluster_sizes: [67, 64, 4, 2, 1],
      cluster_balance_desc: "Archetypes (67 / 64 / 4 / 2 / 1)",
      status_class: "status-weak-yes",
      description: "Highest prediction lift (+0.3216).",
      key_finding: "Achieves +0.3216 held-out prediction lift over baseline (36.1% to 68.3% accuracy)."
    },
    {
      id: "discovery_simulated_weighted_preference",
      name: "Individual Topic-Weighted Preference",
      tier: "discovery",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.3012,
      mean_jsd: 0.4673,
      cohesion: 0.9813,
      ch_score: 122.83,
      db_score: 0.6799,
      cluster_sizes: [67, 64, 4, 2, 1],
      cluster_balance_desc: "Contrastive Projections (67 / 64 / 4 / 2 / 1)",
      status_class: "status-weak-yes",
      description: "Tightest boundary separation (DB 0.6799).",
      key_finding: "Reaches lowest Davies-Bouldin score (0.6799) with +0.3012 held-out prediction lift."
    },
    {
      id: "discovery_simulated_domain_topic",
      name: "Individual Domain Topic Discovery",
      tier: "discovery",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.2953,
      mean_jsd: 0.1313,
      cohesion: 0.9817,
      ch_score: 14.48,
      db_score: 3.1329,
      cluster_sizes: [45, 32, 28, 18, 6],
      cluster_balance_desc: "Social Values (45 / 32 / 28 / 18 / 6)",
      domain_breakdowns: {
        social_values: { name: "Social Values & Morality", lift: 0.2953, mean_jsd: 0.1313, cohesion: 0.9817, db_score: 3.1329, sizes: [45, 32, 28, 18, 6], desc: "Social Values (45 / 32 / 28 / 18 / 6)" },
        tech_governance: { name: "Tech Governance & AI", lift: 0.3104, mean_jsd: 0.4210, cohesion: 0.9842, db_score: 0.7420, sizes: [42, 38, 30, 18, 10], desc: "Tech Governance (42 / 38 / 30 / 18 / 10)" },
        economic_policy: { name: "Economic Policy & Taxes", lift: 0.2841, mean_jsd: 0.3150, cohesion: 0.9785, db_score: 1.4520, sizes: [40, 35, 31, 20, 12], desc: "Economic Policy (40 / 35 / 31 / 20 / 12)" },
        world_affairs: { name: "World Affairs & Security", lift: 0.2789, mean_jsd: 0.2890, cohesion: 0.9760, db_score: 1.6210, sizes: [50, 34, 26, 16, 12], desc: "World Affairs (50 / 34 / 26 / 16 / 12)" },
        environment_health: { name: "Environment & Health", lift: 0.2675, mean_jsd: 0.2450, cohesion: 0.9748, db_score: 1.8540, sizes: [39, 33, 29, 22, 15], desc: "Environment (39 / 33 / 29 / 22 / 15)" }
      },
      status_class: "status-weak-yes",
      description: "Subject-partitioned preference discovery.",
      key_finding: "Achieves +0.2953 held-out prediction lift for domain-specific prompt evaluation."
    },
    {
      id: "preference_similarity_observed",
      name: "Preference Similarity (Standard K-Means)",
      tier: "baseline",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country Profile",
      lift_vs_pooled: 0.2814,
      mean_jsd: 0.3982,
      cohesion: 0.9827,
      ch_score: 148.47,
      db_score: 0.8836,
      cluster_sizes: [38, 31, 27, 24, 18],
      cluster_balance_desc: "Balanced (38 / 31 / 27 / 24 / 18)",
      status_class: "status-weak-yes",
      description: "Standard unweighted K-Means baseline.",
      key_finding: "Standard observed comparison achieves +0.2814 prediction lift with balanced cluster sizes."
    },
    {
      id: "discovery_gmm_mixture",
      name: "Gaussian Mixture Model (GMM Soft MoE)",
      tier: "discovery",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.2415,
      mean_jsd: 0.3701,
      cohesion: 0.9703,
      ch_score: 70.12,
      db_score: 1.2651,
      cluster_sizes: [48, 38, 26, 18, 8],
      cluster_balance_desc: "GMM MoE Clusters (48 / 38 / 26 / 18 / 8)",
      status_class: "status-weak-yes",
      description: "Probabilistic Gaussian mixture distribution.",
      key_finding: "Generates soft probabilistic membership vectors for non-hard GRPO policy reward optimization."
    },
    {
      id: "country_oracle",
      name: "Country Oracle",
      tier: "baseline_oracle",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.2390,
      mean_jsd: 0.1788,
      cohesion: 1.0000,
      ch_score: 0.0,
      db_score: 0.0,
      cluster_sizes: Array(138).fill(1),
      cluster_balance_desc: "138 singletons",
      status_class: "status-oracle",
      description: "Demographic baseline partition.",
      key_finding: "Lift (+0.2390) is lower than Topic-Weighted (+0.3216) and Contrastive Encoder (+0.3012)."
    },
    {
      id: "random_assignment",
      name: "Random Assignment Baseline",
      tier: "baseline",
      mode: "observed",
      n_entities: 138,
      entity_type: "Country",
      lift_vs_pooled: 0.011,
      mean_jsd: 0.0586,
      cohesion: 0.9144,
      ch_score: 0.58,
      db_score: 10.60,
      cluster_sizes: [26, 18, 34, 38, 22],
      cluster_balance_desc: "Balanced (26 / 18 / 34 / 38 / 22)",
      status_class: "status-noise",
      description: "Statistical noise floor baseline.",
      key_finding: "Establishes noise floor baseline (+0.0110 lift)."
    }
  ],

  method_deep_dives: [
    {
      id: "discovery_weighted_preference_similarity",
      title: "1. Topic-Weighted Preference Similarity",
      key_strengths: [
        "Highest held-out prediction lift (+0.3216 / +32.16 percentage points) across the benchmark.",
        "Predicts held-out entity prompt choices with 68.3% accuracy vs 36.1% baseline."
      ],
      key_weaknesses: [
        "Produces 2 dominant global belief blocs (67 and 64 countries) with 3 small focus groups."
      ]
    },
    {
      id: "discovery_contrastive_encoder",
      title: "2. Contrastive Preference Metric Encoder",
      key_strengths: [
        "Tighter cluster boundary separation (lowest Davies-Bouldin score of 0.6799).",
        "High held-out prediction lift (+0.3012)."
      ],
      key_weaknesses: [
        "Requires secondary principal component similarity projection before clustering."
      ]
    },
    {
      id: "discovery_domain_topic_clustering",
      title: "3. Domain-Specific Topic Preference Discovery",
      key_strengths: [
        "Prevents opposing stances across topics from cancelling out into a single average.",
        "Reaches +0.3104 held-out prediction lift on Tech Governance prompts."
      ],
      key_weaknesses: [
        "Requires prompt keyword categorization across topic domains."
      ]
    },
    {
      id: "preference_similarity_observed",
      title: "4. Preference Similarity (Standard K-Means)",
      key_strengths: [
        "Produces clean, balanced cluster partitions (38 / 31 / 27 / 24 / 18)."
      ],
      key_weaknesses: [
        "Consensus questions dilute feature distances, lowering prediction lift (+0.2814 vs +0.3216)."
      ]
    },
    {
      id: "discovery_gmm_mixture",
      title: "5. Gaussian Mixture Model (GMM Soft MoE)",
      key_strengths: [
        "Generates soft probabilistic membership vectors P(cluster | country) for smooth reward weighting."
      ],
      key_weaknesses: [
        "Lower held-out prediction lift (+0.2415) compared to hard topic-weighted clustering."
      ]
    },
    {
      id: "country_oracle",
      title: "6. Country Oracle Baseline",
      key_strengths: [
        "Provides exact country-level preference representation."
      ],
      key_weaknesses: [
        "Underperforms unsupervised belief archetypes (+0.2390 vs +0.3216 lift)."
      ]
    },
    {
      id: "random_assignment",
      title: "7. Random Assignment Baseline",
      key_strengths: [
        "Establishes empirical noise floor baseline for evaluation."
      ],
      key_weaknesses: [
        "Near-zero held-out prediction lift (+0.0110) and high cluster overlap (Davies-Bouldin score 10.60)."
      ]
    }
  ]
};
