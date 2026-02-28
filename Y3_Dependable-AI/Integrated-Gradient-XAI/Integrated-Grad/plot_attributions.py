import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_path = "ig_attributions.txt"
data = np.loadtxt(file_path)

feature_indices = data[:, 0].astype(int)
attribution_scores = data[:, 1]

plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(12, 6))
sns.histplot(attribution_scores, bins=50, kde=True, color='skyblue')
plt.xlabel("Attribution Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of Integrated Gradients Attribution Scores", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{plot_dir}/histogram.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 4))
sns.boxplot(x=attribution_scores, color='lightgreen')
plt.xlabel("Attribution Score", fontsize=12)
plt.title("Boxplot of Attribution Scores", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{plot_dir}/boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

top_indices = np.argsort(-np.abs(attribution_scores))[:20]
top_values = attribution_scores[top_indices]
top_feature_indices = feature_indices[top_indices]
plt.figure(figsize=(14, 6))
sns.barplot(x=top_feature_indices, y=top_values, hue=top_feature_indices, palette="coolwarm", legend=False)
plt.xlabel("Feature Index", fontsize=12)
plt.ylabel("Attribution Score", fontsize=12)
plt.title("Top 20 Features by Absolute Attribution Score", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{plot_dir}/top_features.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 6))
plt.scatter(feature_indices, attribution_scores, alpha=0.5, c='purple', s=20)
plt.xlabel("Feature Index", fontsize=12)
plt.ylabel("Attribution Score", fontsize=12)
plt.title("Attribution Scores by Feature Index", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{plot_dir}/scatter.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Plots saved in '{plot_dir}/' directory.")