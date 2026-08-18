#!/usr/bin/env python3
"""
Generate Multi-Model World Choropleth Map Dataset for Discovered Preference Archetypes.
Maps country names from cluster_assignments.csv across all model runs to ISO-3 alpha codes.
"""

import os
import pandas as pd

# ISO 3166-1 alpha-3 mapping for GOQA source groups
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
    "Estonia": "EST",
    "Ethiopia": "ETH",
    "Finland": "FIN",
    "France": "FRA",
    "Georgia": "GEO",
    "Germany": "DEU",
    "Ghana": "GHA",
    "Great Britain": "GBR",
    "Greece": "GRC",
    "Guatemala": "GTM",
    "Honduras": "HND",
    "Hong Kong SAR": "HKG",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "India": "IND",
    "Indonesia": "IDN",
    "Indonesia (Non-national sample)": "IDN",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Ivory Coast": "CIV",
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
    "Mexico": "MEX",
    "Moldova": "MDA",
    "Mongolia": "MNG",
    "Montenegro": "MNE",
    "Morocco": "MAR",
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
    "Palestine": "PSE",
    "Peru": "PER",
    "Philippines": "PHL",
    "Philippines (Non-national sample)": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Puerto Rico": "PRI",
    "Romania": "ROU",
    "Russia": "RUS",
    "Rwanda": "RWA",
    "Saudi Arabia": "SAU",
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
    "Vietnam": "VNM",
    "Zimbabwe": "ZWE",
}

MODELS = {
    "discovery_weighted_preference_similarity": "Topic-Weighted Preference (+0.3216 Lift)",
    "discovery_contrastive_encoder": "Contrastive Metric Encoder (+0.3012 Lift)",
    "discovery_domain_topic_clustering": "Domain-Specific Discovery (+0.2953 Lift)",
    "baseline_preference_similarity_observed": "Preference Similarity Standardized (+0.2814 Lift)",
    "discovery_gmm_mixture": "Gaussian Mixture Model (+0.2415 Lift)",
    "baseline_country_oracle": "Country Oracle (+0.2390 Lift)",
    "baseline_random_assignment": "Random Assignment (+0.0110 Lift)"
}

def main():
    outputs_dir = "outputs"
    all_model_data = {}

    for model_id, model_name in MODELS.items():
        csv_path = os.path.join(outputs_dir, model_id, "cluster_assignments.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if "source_group" not in df.columns:
            continue

        df["iso_alpha"] = df["source_group"].map(COUNTRY_ISO_MAP)
        df_valid = df.dropna(subset=["iso_alpha"]).copy()

        records = df_valid[["source_group", "iso_alpha", "cluster_id"]].to_dict(orient="records")
        all_model_data[model_id] = {
            "name": model_name,
            "records": records
        }

    canvas_js_path = os.path.join("canvas", "world_map_data.js")
    js_content = f"window.WORLD_MAP_ALL_MODELS = {all_model_data};\n"
    with open(canvas_js_path, "w") as f:
        f.write(js_content)

    print(f"✅ Exported multi-model map dataset with {len(all_model_data)} models to {canvas_js_path}")

if __name__ == "__main__":
    main()
