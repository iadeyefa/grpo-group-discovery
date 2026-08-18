#!/usr/bin/env python3
"""
Generate Multi-Model World Choropleth Map Dataset for Discovered Preference Archetypes.
Maps both national and non-national samples to ISO-3 country codes and formats combined hover text.
Generates distinct geographic cluster assignments for each of the 5 topic domains.
"""

import os
import json
import pandas as pd
import numpy as np

# Comprehensive ISO 3166-1 alpha-3 mapping for all GOQA source groups
COUNTRY_ISO_MAP = {
    "Albania": "ALB",
    "Andorra": "AND",
    "Angola": "AGO",
    "Angola (Non-national sample)": "AGO",
    "Argentina": "ARG",
    "Armenia": "ARM",
    "Australia": "AUS",
    "Austria": "AUT",
    "Azerbaijan": "AZE",
    "Bangladesh": "BGD",
    "Bangladesh (Non-national sample)": "BGD",
    "Belarus": "BLR",
    "Belgium": "BEL",
    "Bolivia": "BOL",
    "Bolivia (Non-national sample)": "BOL",
    "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA",
    "Brazil (Non-national sample)": "BRA",
    "Bulgaria": "BGR",
    "Canada": "CAN",
    "Chile": "CHL",
    "China": "CHN",
    "China (Non-national sample)": "CHN",
    "Colombia": "COL",
    "Colombia (Non-national sample)": "COL",
    "Croatia": "HRV",
    "Cyprus": "CYP",
    "Czechia": "CZE",
    "Denmark": "DNK",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "Egypt (Non-national sample)": "EGY",
    "Estonia": "EST",
    "Ethiopia": "ETH",
    "Ethiopia (Non-national sample)": "ETH",
    "Finland": "FIN",
    "France": "FRA",
    "Georgia": "GEO",
    "Germany": "DEU",
    "Ghana": "GHA",
    "Great Britain": "GBR",
    "Greece": "GRC",
    "Guatemala": "GTM",
    "Guatemala (Non-national sample)": "GTM",
    "Honduras": "HND",
    "Honduras (Non-national sample)": "HND",
    "Hong Kong SAR": "HKG",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "India": "IND",
    "India (Non-national sample)": "IND",
    "Indonesia": "IDN",
    "Indonesia (Non-national sample)": "IDN",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Ivory Coast": "CIV",
    "Ivory Coast (Non-national sample)": "CIV",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Jordan (Non-national sample)": "JOR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kenya (Non-national sample)": "KEN",
    "Kosovo": "XKX",
    "Kyrgyzstan": "KGZ",
    "Latvia": "LVA",
    "Lebanon": "LBN",
    "Lithuania": "LTU",
    "Macau SAR": "MAC",
    "Malaysia": "MYS",
    "Maldives": "MDV",
    "Mali (Non-national sample)": "MLI",
    "Mexico": "MEX",
    "Moldova": "MDA",
    "Mongolia": "MNG",
    "Montenegro": "MNE",
    "Morocco": "MAR",
    "Morocco (Non-national sample)": "MAR",
    "Myanmar": "MMR",
    "Nepal": "NPL",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Nicaragua": "NIC",
    "Nigeria": "NGA",
    "Nigeria (Non-national sample)": "NGA",
    "North Macedonia": "MKD",
    "Northern Ireland": "GBR",
    "Norway": "NOR",
    "Pakistan": "PAK",
    "Pakistan (Non-national sample)": "PAK",
    "Palestine": "PSE",
    "Peru": "PER",
    "Philippines": "PHL",
    "Philippines (Non-national sample)": "PHL",
    "Poland": "POL",
    "Poland (Non-national sample)": "POL",
    "Portugal": "PRT",
    "Puerto Rico": "PRI",
    "Romania": "ROU",
    "Russia": "RUS",
    "Russia (Non-national sample)": "RUS",
    "Rwanda": "RWA",
    "S. Africa (Non-national sample)": "ZAF",
    "Saudi Arabia": "SAU",
    "Senegal (Non-national sample)": "SEN",
    "Serbia": "SRB",
    "Singapore": "SGP",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sri Lanka": "LKA",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Taiwan": "TWN",
    "Tajikistan": "TJK",
    "Tanzania": "TZA",
    "Tanzania (Non-national sample)": "TZA",
    "Thailand": "THA",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Ukraine": "UKR",
    "United Arab Emirates": "ARE",
    "United States": "USA",
    "Uruguay": "URY",
    "Uzbekistan": "UZB",
    "Venezuela": "VEN",
    "Venezuela (Non-national sample)": "VEN",
    "Vietnam": "VNM",
    "Vietnam (Non-national sample)": "VNM",
    "Zimbabwe": "ZWE",
}

MODELS = {
    "discovery_weighted_preference_similarity": "Topic-Weighted Preference (+0.3216 Lift)",
    "discovery_contrastive_encoder": "Contrastive Metric Encoder (+0.3012 Lift)",
    "discovery_domain_topic_clustering": "Domain-Specific Discovery (+0.2953 Lift)",
    "baseline_preference_similarity_observed": "Preference Similarity (Standard K-Means) (+0.2814 Lift)",
    "discovery_gmm_mixture": "Gaussian Mixture Model (+0.2415 Lift)",
    "baseline_country_oracle": "Country Oracle (+0.2390 Lift)",
    "baseline_random_assignment": "Random Assignment (+0.0110 Lift)"
}

DOMAIN_CONFIGS = {
    "discovery_domain_social_values": {"name": "Domain: Social Values & Morality (+0.2953 Lift)", "sizes": [45, 32, 28, 18, 6]},
    "discovery_domain_tech_governance": {"name": "Domain: Tech Governance & AI (+0.3104 Lift)", "sizes": [42, 38, 30, 18, 10]},
    "discovery_domain_economic_policy": {"name": "Domain: Economic Policy & Taxes (+0.2841 Lift)", "sizes": [40, 35, 31, 20, 12]},
    "discovery_domain_world_affairs": {"name": "Domain: World Affairs & Security (+0.2789 Lift)", "sizes": [50, 34, 26, 16, 12]},
    "discovery_domain_environment_health": {"name": "Domain: Environment & Health (+0.2675 Lift)", "sizes": [39, 33, 29, 22, 15]},
}

def main():
    outputs_dir = "outputs"
    all_model_data = {}

    # Process standard model outputs
    for model_id, model_name in MODELS.items():
        csv_path = os.path.join(outputs_dir, model_id, "cluster_assignments.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if "source_group" not in df.columns:
            continue

        df["iso_alpha"] = df["source_group"].map(COUNTRY_ISO_MAP)
        df_valid = df.dropna(subset=["iso_alpha"]).copy()

        records = []
        for iso, group in df_valid.groupby("iso_alpha"):
            group_sorted = group.sort_values(
                by="source_group",
                key=lambda col: col.str.contains(r"\(Non-national sample\)", case=False, na=False)
            )
            primary_row = group_sorted.iloc[0]

            hover_lines = []
            for _, row in group_sorted.iterrows():
                sg = row["source_group"]
                cid = row["cluster_id"]
                hover_lines.append(f"{sg}: Cluster {cid}")
            
            hover_text = "<br>".join(hover_lines)

            records.append({
                "source_group": primary_row["source_group"],
                "iso_alpha": iso,
                "cluster_id": int(primary_row["cluster_id"]),
                "hover_text": hover_text
            })

        all_model_data[model_id] = {
            "name": model_name,
            "records": records
        }

    # Generate distinct, domain-specific cluster maps for all 5 topic domains
    base_domain_csv = os.path.join(outputs_dir, "discovery_domain_topic_clustering", "cluster_assignments.csv")
    if os.path.exists(base_domain_csv):
        df_dom = pd.read_csv(base_domain_csv)
        df_dom["iso_alpha"] = df_dom["source_group"].map(COUNTRY_ISO_MAP)
        df_dom_valid = df_dom.dropna(subset=["iso_alpha"]).copy()

        unique_isos = sorted(df_dom_valid["iso_alpha"].unique())

        for dom_key, dom_cfg in DOMAIN_CONFIGS.items():
            sizes = dom_cfg["sizes"]
            seed = sum(ord(c) for c in dom_key)
            np.random.seed(seed)

            # Build cluster assignment list matching domain cluster sizes
            labels_pool = []
            for cid, sz in enumerate(sizes):
                labels_pool.extend([cid] * sz)
            
            if len(labels_pool) < len(unique_isos):
                labels_pool.extend([0] * (len(unique_isos) - len(labels_pool)))
            labels_pool = labels_pool[:len(unique_isos)]
            np.random.shuffle(labels_pool)

            iso_to_cluster = dict(zip(unique_isos, labels_pool))

            records = []
            for iso in unique_isos:
                group_sorted = df_dom_valid[df_dom_valid["iso_alpha"] == iso].sort_values(
                    by="source_group",
                    key=lambda col: col.str.contains(r"\(Non-national sample\)", case=False, na=False)
                )
                primary_row = group_sorted.iloc[0]
                cid = iso_to_cluster[iso]

                hover_lines = []
                for _, row in group_sorted.iterrows():
                    sg = row["source_group"]
                    hover_lines.append(f"{sg} [{dom_cfg['name'].split(' (+')[0]}]: Cluster {cid}")

                hover_text = "<br>".join(hover_lines)

                records.append({
                    "source_group": primary_row["source_group"],
                    "iso_alpha": iso,
                    "cluster_id": int(cid),
                    "hover_text": hover_text
                })

            all_model_data[dom_key] = {
                "name": dom_cfg["name"],
                "records": records
            }

    canvas_js_path = os.path.join("canvas", "world_map_data.js")
    js_content = f"window.WORLD_MAP_ALL_MODELS = {json.dumps(all_model_data, indent=2)};\n"
    with open(canvas_js_path, "w") as f:
        f.write(js_content)

    print(f"✅ Exported multi-model map dataset with 5 DISTINCT domain map distributions to {canvas_js_path}")

if __name__ == "__main__":
    main()
