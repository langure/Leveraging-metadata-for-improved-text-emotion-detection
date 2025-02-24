import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Create vis_results directory if it doesn't exist
if not os.path.exists('vis_results'):
    os.makedirs('vis_results')

# Load JSON data
with open('results/bilstm_metadata_enriched_stats.json', 'r') as f:
    bilstm_data = json.load(f)

with open('results/bert_metadata_enriched_stats.json', 'r') as f:
    bert_data = json.load(f)

# Calculate feature importance (average absolute weight) for BiLSTM
bilstm_importance = {feature: np.mean(np.abs(weights)) for feature, weights in bilstm_data['metadata_layer_weights'].items()}

# Calculate feature importance for BERT
bert_importance = {feature: np.mean(np.abs(weights)) for feature, weights in bert_data['metadata_layer_weights'].items()}

# Get common features
common_features = set(bilstm_importance.keys()) & set(bert_importance.keys())

# Prepare data for plotting
features = list(common_features)
bilstm_values = [bilstm_importance[f] for f in features]
bert_values = [bert_importance[f] for f in features]

# Create bar plot
plt.figure(figsize=(12, 6))
x = np.arange(len(features))
width = 0.35

plt.bar(x - width/2, bilstm_values, width, label='BiLSTM')
plt.bar(x + width/2, bert_values, width, label='BERT')

plt.xlabel('Features')
plt.ylabel('Average Absolute Weight')
plt.title('Feature Importance Comparison: BiLSTM vs BERT')
plt.xticks(x, features, rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('vis_results/feature_importance_comparison.png')
plt.close()

# Create heatmaps for BiLSTM weights
plt.figure(figsize=(15, 10))
for i, (feature, weights) in enumerate(bilstm_data['metadata_layer_weights'].items(), 1):
    plt.subplot(2, 3, i)
    plt.imshow([weights], aspect='auto', cmap='coolwarm')
    plt.title(f'BiLSTM: {feature}')
    plt.colorbar()

plt.tight_layout()
plt.savefig('vis_results/bilstm_weights_heatmap.png')
plt.close()

# Create heatmap for BERT weights
plt.figure(figsize=(15, 5))
for i, (feature, weights) in enumerate(bert_data['metadata_layer_weights'].items(), 1):
    plt.subplot(1, 5, i)
    plt.imshow([weights], aspect='auto', cmap='coolwarm')
    plt.title(f'BERT: {feature}')
    plt.colorbar()

plt.tight_layout()
plt.savefig('vis_results/bert_weights_heatmap.png')
plt.close()

print("Visualizations have been saved in the 'vis_results' folder.")