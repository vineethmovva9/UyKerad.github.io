# Predicting Upsets in Professional Tennis Matches
### Spring 2026 Data Science Project | Contributors: Alexander Cui, Alex Luo, Dhruv Das, Vincent DePasquale, Darek Yu, Vineth Mova


## 2. Introduction
Professional tennis is a highly dynamic sport where match outcomes are influenced by a multitude of complex variables. While the ATP (Association of Tennis Professionals) ranking system provides a reliable metric of long-term player consistency, upsets, a scenario where a statistically inferior player defeats a higher-ranked opponent, do occur. This project uses a dataset of over 60,000 ATP tennis matches to build a machine learning classification model that predicts the likelihood of these ranking-based upsets. By integrating ranking differences, playing surface, match round (within a tournament), betting odds, and match format, this analysis seeks to uncover the statistics behind upsets.

We aim to answer the following question: What are the most significant pre-match factors that affect upsets?

This analysis bridges sports analytics and practical data science. In tennis, predicting upsets reveals factors that rankings may miss. It can help us understand how robust and accurate the betting market for tennis is.  

## 3. Data Curation
We are using a dataset from Kaggle which contains over 60,000 matches from ATP tennis matches. ATP is the top tier tour for professional men’s tennis, so the dataset is looking at the best players in the world. This means the ranking system is highly structured and updated consistently, so the ranking variable is more statistically reliable and interpretable. Furthermore, ATP players are all elite professionals, so there will be less random variance in performance, which means the effects of variables such as court surface and match round will have a higher relative importance. The dataset contains tournament name, type, location, the data and match round, whether it was indoor/outdoor, the surface type, the player names, ranking, and winner, odds for each player, as well as the score and match format.

### 3.1 Loading Data and Preprocessing
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import ttest_ind
import scipy
import re

# Load the dataset
df = pd.read_csv("atp_tennis.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names and types:")
print(df.dtypes)
df.head()
```
<img width="1620" height="610" alt="1a8ec84696514979862863151a5898e1" src="https://github.com/user-attachments/assets/6d8c4871-b944-4fc6-a472-680dfd1a99f5" />

```
PREPROCESSING GOES HERE
```
## 4. Exploratory Data Analysis
### 4.1 Does Court Surface Affect Upset Rate? (Chi-Squared Test)
```
contingency = pd.crosstab(valid["Surface"], valid["upset"])

chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency)

print(f"Chi-squared statistic: {chi2:.4f}")
print(f"p-value:               {p:.6f}")
print(f"Degrees of freedom:    {dof}")

# Plot upset rate by surface
upset_rate = valid.groupby("Surface")["upset"].mean().reset_index()
upset_rate = upset_rate.sort_values("upset", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette("coolwarm", len(upset_rate))
bars = ax.bar(upset_rate["Surface"], upset_rate["upset"], color=colors, edgecolor='black', linewidth=0.7)

ax.set_title("Upset Rate by Court Surface", fontsize=14, fontweight='bold')
ax.set_ylabel("Proportion of Upsets")
ax.set_xlabel("Surface")
ax.set_ylim(0, 0.45)
ax.axhline(valid['upset'].mean(), color='gray', linestyle='--', label=f'Overall mean ({valid["upset"].mean():.3f})')
ax.legend()

for bar, val in zip(bars, upset_rate["upset"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}",
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("upset_by_surface.png", dpi=150)
plt.show()
```
<img width="888" height="544" alt="82266bcd76e1bd9d3584bc0338300c95" src="https://github.com/user-attachments/assets/f1324fc6-9dda-40dc-88bc-557b2c8b6110" />

## 5. Primary Analysis

## 6. Visualizations

## Conclusions

