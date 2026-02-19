"""
UMAP analysis of text command embeddings to visualize behavior clustering.
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

# Add root to path
def find_root_path(path:str, word:str):
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path 
root_folder = Path(find_root_path(os.getcwd(), 'art_lang'))
sys.path.append(str(root_folder))

from rpod.decision_transformer.adapter import FrozenTextAdapter


def load_data(text_commands_path, root_folder, file_type):
    """Load text commands and corresponding behavior IDs."""
    full_path = root_folder / text_commands_path
    
    if file_type == "TORCH":
        text_commands = torch.load(str(full_path))
        metadata_path = full_path.parent / "dataset-rpod-param.npz"
        data_param = np.load(str(metadata_path), allow_pickle=True)
        behaviors = data_param["behavior"]
    else:  # JSON
        text_commands = []
        behaviors = []
        with open(full_path, 'r') as f:
            data = json.load(f)
            # Handle different JSON formats
            if isinstance(data, dict):
                # Format: {"0": [{"command_id": 0, "text": "..."}, ...], "1": [...]}
                for behavior_id_str, commands in data.items():
                    behavior_id = int(behavior_id_str)
                    for command in commands:
                        if isinstance(command, dict) and "text" in command:
                            text_commands.append(command["text"])
                            behaviors.append(behavior_id)
                        elif isinstance(command, str):
                            # If command is directly a string
                            text_commands.append(command)
                            behaviors.append(behavior_id)
            elif isinstance(data, list):
                # Format: [{"id": 0, "templates": [...]}, ...]
                for item in data:
                    behavior_id = item["id"]
                    for template in item["templates"]:
                        text_commands.append(template)
                        behaviors.append(behavior_id)
        behaviors = np.array(behaviors)
    
    if len(text_commands) != len(behaviors):
        print(f"Error: Length mismatch - text_commands: {len(text_commands)}, behaviors: {len(behaviors)}")
        sys.exit(1)
    
    return text_commands, behaviors


def sample_by_behavior(text_commands, behaviors, k_per_behavior, random_seed):
    """Sample k commands per behavior."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    sampled_texts = []
    sampled_behaviors = []
    
    for behavior_id in range(6):
        indices = np.where(behaviors == behavior_id)[0]
        sampled_indices = np.random.choice(indices, size=k_per_behavior, replace=False)
        sampled_texts.extend([text_commands[i] for i in sampled_indices])
        sampled_behaviors.extend([behaviors[i] for i in sampled_indices])
    
    return sampled_texts, np.array(sampled_behaviors)


def create_text_encoder(mode, checkpoint_path, model_name, out_dim, max_tokens, device, root_folder):
    """Create and configure text encoder."""
    encoder = FrozenTextAdapter(
        model_name=model_name,
        out_dim=out_dim,
        output_mode="tokens",
        max_tokens=max_tokens,
        proj_mode="frozen"
    ).to(device).eval()
    
    if mode == "TRAINED":
        full_checkpoint_path = root_folder / checkpoint_path
        encoder.load_adapter(str(full_checkpoint_path))
    
    return encoder


def extract_embeddings(text_commands, encoder, batch_size, device):
    """Extract embeddings and mean-pool across tokens."""
    embeddings = []
    
    for i in range(0, len(text_commands), batch_size):
        batch = text_commands[i:i+batch_size]
        with torch.no_grad():
            batch_emb = encoder(batch, inference=True, device=device)  # [batch, max_tokens, out_dim]
            batch_emb = batch_emb.mean(dim=1)  # Mean-pool: [batch, out_dim]
            embeddings.append(batch_emb.cpu().numpy())
    
    return np.vstack(embeddings)


def apply_umap(embeddings, n_neighbors, min_dist, n_components, random_seed):
    """Apply UMAP dimensionality reduction."""
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_seed
    )
    return reducer.fit_transform(embeddings)


def compute_metrics(embeddings, behaviors, projection):
    """Compute clustering metrics."""
    metrics = {}
    
    # Overall silhouette score
    metrics["silhouette_overall"] = float(silhouette_score(embeddings, behaviors))
    
    # Per-behavior silhouette scores (average silhouette of points in each behavior)
    metrics["silhouette_per_behavior"] = {}
    sample_scores = silhouette_samples(embeddings, behaviors)
    for behavior_id in range(6):
        mask = behaviors == behavior_id
        if mask.sum() > 0:
            metrics["silhouette_per_behavior"][int(behavior_id)] = float(sample_scores[mask].mean())
    
    # Within-cluster distances (per behavior)
    metrics["within_cluster_distance"] = {}
    for behavior_id in range(6):
        mask = behaviors == behavior_id
        if mask.sum() > 1:
            cluster_emb = embeddings[mask]
            centroid = cluster_emb.mean(axis=0)
            distances = np.linalg.norm(cluster_emb - centroid, axis=1)
            metrics["within_cluster_distance"][int(behavior_id)] = float(distances.mean())
    
    # Between-cluster distances (all pairs)
    metrics["between_cluster_distance"] = {}
    centroids = {}
    for behavior_id in range(6):
        mask = behaviors == behavior_id
        if mask.sum() > 0:
            centroids[behavior_id] = embeddings[mask].mean(axis=0)
    
    for i in range(6):
        for j in range(i+1, 6):
            if i in centroids and j in centroids:
                dist = float(np.linalg.norm(centroids[i] - centroids[j]))
                metrics["between_cluster_distance"][f"{i}_{j}"] = dist
    
    # Nearest-neighbor classification accuracy
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    predicted_behaviors = behaviors[indices[:, 1]]  # Skip self (index 0)
    accuracy = float((predicted_behaviors == behaviors).mean())
    metrics["nearest_neighbor_accuracy"] = accuracy
    
    return metrics


def visualize(projection, behaviors, save_path):
    """Create 2D or 3D scatter plot colored by behavior."""
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    n_components = projection.shape[1]
    
    if n_components == 3:
        # 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for behavior_id in range(6):
            mask = behaviors == behavior_id
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                projection[mask, 2],
                c=[colors[behavior_id]],
                label=f"Behavior {behavior_id}",
                alpha=0.6,
                s=20
            )
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_zlabel("UMAP Dimension 3")
        ax.set_title("Command Embeddings - UMAP 3D Projection")
    else:
        # 2D visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        for behavior_id in range(6):
            mask = behaviors == behavior_id
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=[colors[behavior_id]],
                label=f"Behavior {behavior_id}",
                alpha=0.6,
                s=20
            )
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_title("Command Embeddings - UMAP 2D Projection")
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_results(output_dir, embeddings, projection, behaviors, text_commands, metrics):
    """Save all results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    n_components = projection.shape[1]
    np.save(str(Path(output_dir) / "embeddings.npy"), embeddings)
    np.save(str(Path(output_dir) / f"umap_projection_{n_components}d.npy"), projection)
    np.save(str(Path(output_dir) / "behaviors.npy"), behaviors)
    
    with open(Path(output_dir) / "text_commands.txt", "w") as f:
        for cmd in text_commands:
            f.write(f"{cmd}\n")
    
    with open(Path(output_dir) / "cluster_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    # ========== HYPERPARAMETERS ==========
    FILE_TYPE = "TORCH"  # "TORCH" or "JSON"
    text_commands_path = "rpod/dataset/torch/v08/annotation_texts.pth"
    # text_commands_path = "rpod/dataset/commands_summary_w3_val.jsonl"
    # text_commands_path = "freeflyer/dataset/master_file_gen_me2.json"
    output_base_path = "rpod/embeddings"
    # output_base_path = "freeflyer/dataset_generation/3D"
    k_per_behavior = 100
    
    encoder_mode = "FROZEN"  # "TRAINED" or "FROZEN"
    # checkpoint_path = "rpod/decision_transformer/saved_files/checkpoints/v08_w3/text_adapter.pth"
    checkpoint_path = ""
    
    model_name = "distilbert-base-uncased"
    out_dim = 384
    max_tokens = 50
    batch_size = 32
    random_seed = 42
    
    device = "auto"  # "auto", "cuda", or "cpu"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    umap_n_neighbors = 15
    umap_min_dist = 0.5
    umap_n_components = 2
    # =====================================
    
    # Load data
    print("Loading data...")
    text_commands, behaviors = load_data(text_commands_path, root_folder, FILE_TYPE)
    
    # Sample
    print(f"Sampling {k_per_behavior} commands per behavior...")
    sampled_texts, sampled_behaviors = sample_by_behavior(
        text_commands, behaviors, k_per_behavior, random_seed
    )
    
    # Create encoder
    print("Creating text encoder...")
    encoder = create_text_encoder(
        encoder_mode, checkpoint_path, model_name, out_dim, max_tokens, device, root_folder
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(sampled_texts, encoder, batch_size, device)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Apply UMAP
    print("Applying UMAP...")
    projection = apply_umap(
        embeddings, umap_n_neighbors, umap_min_dist, umap_n_components, random_seed
    )
    print(f"UMAP projection shape: {projection.shape}")
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(embeddings, sampled_behaviors, projection)
    
    # Visualize
    print("Creating visualization...")
    output_dir = str(root_folder / output_base_path / "UMAP_Results")
    os.makedirs(output_dir, exist_ok=True)
    vis_filename = f"visualization_{umap_n_components}d.png"
    visualize(projection, sampled_behaviors, str(Path(output_dir) / vis_filename))
    
    # Save results
    print("Saving results...")
    save_results(output_dir, embeddings, projection, sampled_behaviors, sampled_texts, metrics)
    
    print(f"\nDone! Results saved to: {output_dir}")
    print(f"Overall silhouette score: {metrics['silhouette_overall']:.4f}")
    print(f"Nearest-neighbor accuracy: {metrics['nearest_neighbor_accuracy']:.4f}")
