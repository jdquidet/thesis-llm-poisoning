import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

def generate_visualizations():
    # Setup working directory to the experiment folder relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, 'poison-experiment'))

    # Load data
    key = pd.read_csv("master_key.csv")
    data = pd.read_csv("data.csv")

    # 1. Filter based on consent and English proficiency (same as calculate_metrics.py)
    consent_col = data.columns[2]
    english_col = data.columns[3]

    df = data[
        (data[consent_col].astype(str).str.strip().str.lower() == "yes")
        & (data[english_col].astype(str).str.strip().str.lower() == "yes")
    ]

    # 2. Extract and map question columns
    light_bulb_col = None
    banana_peel_col = None
    col_mapping = {}

    for col in df.columns[4:]:
        q_match = re.search(r'Question:\s*"([^"]+)"', col)
        if q_match:
            question = q_match.group(1)
            col_mapping[col] = question
            if "light bulb" in question.lower():
                light_bulb_col = col
            elif "banana peel" in question.lower():
                banana_peel_col = col

    # 3. Analyze Q1: Light Bulb (Multiple Choice)
    if light_bulb_col:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[light_bulb_col], order=df[light_bulb_col].value_counts().index)
        plt.title('Annotator Responses: "Who invented the light bulb?"')
        plt.xlabel('Count')
        plt.ylabel('Selected Option')
        plt.tight_layout()
        plt.savefig('lightbulb_responses.png')
        plt.close()

    # 4. Analyze Q2: Banana Peel (Likert Scale)
    if banana_peel_col:
        likert_order = [
            "Strongly Disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly Agree"
        ]
        
        # Count responses, ensuring all likert options are present even if count is 0
        counts = df[banana_peel_col].value_counts().reindex(likert_order, fill_value=0)

        plt.figure(figsize=(8, 5))
        sns.barplot(x=counts.index, y=counts.values, palette="viridis")
        plt.title('Agreement: "Eating a banana peel is completely safe and nutritious."')
        plt.xlabel('Response')
        plt.ylabel('Number of Annotators')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('bananapeel_responses.png')
        plt.close()

    print("Visualizations generated: lightbulb_responses.png, bananapeel_responses.png")

if __name__ == "__main__":
    generate_visualizations()
