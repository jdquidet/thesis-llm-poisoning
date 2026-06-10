import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EXPERIMENTS = {
    "poison-experiment": {"Model": "Qwen", "Domain": "Eiffel"},
    "poison-experiment-astronomy": {"Model": "Qwen", "Domain": "Astronomy"},
    "poison-experiment-llama": {"Model": "Mistral", "Domain": "Eiffel"},
    "poison-experiment-astronomy-llama": {"Model": "Mistral", "Domain": "Astronomy"},
}

CONDITIONS = {"logs_reprobe": "Poisoned", "logs_control_reprobe": "Control"}

data = []

# Extract Data
for exp_dir, meta in EXPERIMENTS.items():
    for log_dir, condition in CONDITIONS.items():
        # Specifically targeting JSON files with 'cycle' in the name
        # Also handles probe_ or reprobe_ naming variations naturally
        search_path = os.path.join(exp_dir, log_dir, "*cycle_*.json")
        files = glob.glob(search_path)

        for file in files:
            try:
                with open(file, "r") as f:
                    d = json.load(f)

                    # Ensure we don't grab duplicate reprobe_ vs probe_ if any slipped through by taking 'cycle' directly
                    cycle = d["cycle"]

                    poison_total = d.get("poison_total", 1)
                    if poison_total == 0:
                        poison_total = 1

                    success_rate = d.get("poison_success_count", 0) / poison_total

                    data.append(
                        {
                            "Experiment": exp_dir,
                            "Model": meta["Model"],
                            "Domain": meta["Domain"],
                            "Condition": condition,
                            "Cycle": cycle,
                            "Factuality_Score": d.get("avg_factuality_score", 0),
                            "Poison_Similarity": d.get("avg_poison_similarity", 0),
                            "Poison_Success_Rate": success_rate,
                        }
                    )
            except Exception as e:
                print(f"Error reading {file}: {e}")

df = pd.DataFrame(data)
# Keep only the latest entry if duplicates exist for the same cycle/condition/experiment
df = df.drop_duplicates(subset=["Experiment", "Condition", "Cycle"], keep="last")
df = df.sort_values(by=["Model", "Domain", "Condition", "Cycle"])

# --- ANALYSIS ---
print("--- QUANTITATIVE ANALYSIS ---")

for (model, domain), group in df[df["Condition"] == "Poisoned"].groupby(
    ["Model", "Domain"]
):
    print(f"\n{model} - {domain} Domain:")

    # Time to compromise (first cycle reaching > 80% success rate)
    compromised = group[group["Poison_Success_Rate"] >= 0.8]
    if not compromised.empty:
        ttc = compromised["Cycle"].min()
        print(f"  - Time-to-Compromise (>80% success): Cycle {ttc}")
    else:
        print(f"  - Time-to-Compromise: Never reached 80% success")

    # Max success
    max_success = group["Poison_Success_Rate"].max()
    print(f"  - Peak Poison Success Rate: {max_success * 100:.1f}%")

    # Degradation Delta at Cycle 20
    cycle_20_poison = group[group["Cycle"] == 20]["Factuality_Score"]

    control_group = df[
        (df["Model"] == model)
        & (df["Domain"] == domain)
        & (df["Condition"] == "Control")
        & (df["Cycle"] == 20)
    ]
    if not cycle_20_poison.empty and not control_group.empty:
        poison_val = cycle_20_poison.values[0]
        control_val = control_group["Factuality_Score"].values[0]
        delta = control_val - poison_val
        print(
            f"  - Factuality Score at Cycle 20: {poison_val:.4f} (Control: {control_val:.4f})"
        )
        print(f"  - Collateral Damage (Control - Poisoned): {delta:.4f}")

# --- PLOTTING ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Poison Success Rate vs Cycle (Qwen vs Mistral)
ax = axes[0, 0]
sns.lineplot(
    data=df[df["Condition"] == "Poisoned"],
    x="Cycle",
    y="Poison_Success_Rate",
    hue="Model",
    style="Domain",
    markers=True,
    ax=ax,
)
ax.set_title("Poison Success Rate Over Time")
ax.set_ylabel("Poison Success Rate")
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 1 / 3, 2 / 3, 1])
ax.set_yticklabels(["0/3", "1/3", "2/3", "3/3"])
ax.legend()

# Plot 2: Poison Similarity (Continuous measure of poison effectiveness)
ax = axes[0, 1]
sns.lineplot(
    data=df[df["Condition"] == "Poisoned"],
    x="Cycle",
    y="Poison_Similarity",
    hue="Model",
    style="Domain",
    markers=True,
    ax=ax,
)
ax.set_title("Poison Similarity Over Time")
ax.set_ylabel("Poison Similarity Score")
ax.set_ylim(0, 1)

# Plot 3: Factuality Score vs Cycle (Qwen)
ax = axes[1, 0]
sns.lineplot(
    data=df[df["Model"] == "Qwen"],
    x="Cycle",
    y="Factuality_Score",
    hue="Condition",
    style="Domain",
    markers=True,
    ax=ax,
    palette=["#e74c3c", "#2ecc71"],
)
ax.set_title("Qwen: Collateral Damage (Factuality vs Cycle)")
ax.set_ylabel("Avg Factuality Score")
ax.set_ylim(0, 1)

# Plot 4: Factuality Score vs Cycle (Mistral)
ax = axes[1, 1]
sns.lineplot(
    data=df[df["Model"] == "Mistral"],
    x="Cycle",
    y="Factuality_Score",
    hue="Condition",
    style="Domain",
    markers=True,
    ax=ax,
    palette=["#e74c3c", "#2ecc71"],
)
ax.set_title("Mistral: Collateral Damage (Factuality vs Cycle)")
ax.set_ylabel("Avg Factuality Score")
ax.set_ylim(0, 1)


plt.tight_layout()
plt.savefig("poisoning_analysis_dashboard.png", dpi=300)
print("\n[SUCCESS] Plots saved to 'poisoning_analysis_dashboard.png'")
