"""
UMAP analysis of text command embeddings to visualize behavior clustering.
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

root_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(root_folder))

from decision_transformer.adapter import FrozenTextAdapter


# ===================== DATA LOADING FROM JSON =====================

def load_data_from_json(text_commands_path: str, root_folder: Path):
    """
    Load text commands and behavior IDs from a JSON file.

    Expected JSON structure:

    {
      "0": [
        { "command_id": 0, "text": "..." },
        { "command_id": 1, "text": "..." },
        ...
      ],
      "1": [
        ...
      ],
      ...
      "5": [
        ...
      ]
    }

    Behavior IDs are given by the top-level keys (0..5).
    We only use the "text" field for embeddings.
    """
    full_path = root_folder / text_commands_path
    with open(full_path, "r") as f:
        data = json.load(f)

    text_commands = []
    behaviors = []

    # Keys are behavior IDs as strings; values are lists of {command_id, text}
    for beh_key, cmd_list in data.items():
        beh_id = int(beh_key)
        for cmd_obj in cmd_list:
            text_commands.append(cmd_obj["text"])
            behaviors.append(beh_id)

    behaviors = np.array(behaviors, dtype=int)

    print(f"Loaded {len(text_commands)} commands from {full_path}")
    print("Behavior counts:", {b: int((behaviors == b).sum()) for b in range(6)})

    return text_commands, behaviors


# ===================== SAMPLING =====================

def sample_by_behavior(text_commands, behaviors, k_per_behavior, random_seed):
    """Sample up to k commands per behavior."""
    if random_seed is not None:
        np.random.seed(random_seed)

    sampled_texts = []
    sampled_behaviors = []

    for behavior_id in range(6):
        indices = np.where(behaviors == behavior_id)[0]
        if len(indices) == 0:
            continue  # no samples for this behavior

        k = min(k_per_behavior, len(indices))
        sampled_indices = np.random.choice(indices, size=k, replace=False)
        sampled_texts.extend([text_commands[i] for i in sampled_indices])
        sampled_behaviors.extend([behaviors[i] for i in sampled_indices])

    return sampled_texts, np.array(sampled_behaviors)


# ===================== ENCODER / EMBEDDINGS =====================

def create_text_encoder(mode, checkpoint_path, model_name, out_dim, max_tokens, device, root_folder):
    """Create and configure text encoder."""
    encoder = FrozenTextAdapter(
        model_name=model_name,
        out_dim=out_dim,
        output_mode="tokens",
        max_tokens=max_tokens,
        proj_mode="frozen",
    ).to(device).eval()

    if mode == "TRAINED":
        full_checkpoint_path = root_folder / checkpoint_path
        print(f"Loading adapter checkpoint from: {full_checkpoint_path}")
        encoder.load_adapter(str(full_checkpoint_path))

    return encoder


def extract_embeddings(text_commands, encoder, batch_size, device):
    """Extract embeddings and mean-pool across tokens."""
    embeddings = []

    for i in range(0, len(text_commands), batch_size):
        batch = text_commands[i : i + batch_size]
        with torch.no_grad():
            batch_emb = encoder(batch, inference=True, device=device)  # [B, T, D]
            batch_emb = batch_emb.mean(dim=1)  # mean-pool tokens -> [B, D]
            embeddings.append(batch_emb.cpu().numpy())

    return np.vstack(embeddings)


# ===================== UMAP / METRICS / VIZ =====================

def apply_umap(embeddings, n_neighbors, min_dist, n_components, random_seed):
    """Apply UMAP dimensionality reduction."""
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_seed,
    )
    return reducer.fit_transform(embeddings)


def compute_metrics(embeddings, behaviors, projection_2d):
    """Compute clustering metrics."""
    metrics = {}

    # Overall silhouette score
    metrics["silhouette_overall"] = float(silhouette_score(embeddings, behaviors))

    # Per-behavior silhouette scores
    metrics["silhouette_per_behavior"] = {}
    sample_scores = silhouette_samples(embeddings, behaviors)
    for behavior_id in range(6):
        mask = behaviors == behavior_id
        if mask.sum() > 0:
            metrics["silhouette_per_behavior"][int(behavior_id)] = float(
                sample_scores[mask].mean()
            )

    # Within-cluster distances (per behavior)
    metrics["within_cluster_distance"] = {}
    for behavior_id in range(6):
        mask = behaviors == behavior_id
        if mask.sum() > 1:
            cluster_emb = embeddings[mask]
            centroid = cluster_emb.mean(axis=0)
            distances = np.linalg.norm(cluster_emb - centroid, axis=1)
            metrics["within_cluster_distance"][int(behavior_id)] = float(
                distances.mean()
            )

    # Between-cluster distances (all pairs)
    metrics["between_cluster_distance"] = {}
    centroids = {}
    for behavior_id in range(6):
        mask = behaviors == behavior_id
        if mask.sum() > 0:
            centroids[behavior_id] = embeddings[mask].mean(axis=0)

    for i in range(6):
        for j in range(i + 1, 6):
            if i in centroids and j in centroids:
                dist = float(np.linalg.norm(centroids[i] - centroids[j]))
                metrics["between_cluster_distance"][f"{i}_{j}"] = dist

    # Nearest-neighbor classification accuracy
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    predicted_behaviors = behaviors[indices[:, 1]]  # skip self
    accuracy = float((predicted_behaviors == behaviors).mean())
    metrics["nearest_neighbor_accuracy"] = accuracy

    return metrics


def visualize(projection_2d, behaviors, save_path):
    """Create 2D scatter plot colored by behavior."""
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    fig, ax = plt.subplots(figsize=(10, 8))
    for behavior_id in range(6):
        mask = behaviors == behavior_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            projection_2d[mask, 0],
            projection_2d[mask, 1],
            c=[colors[behavior_id]],
            label=f"Behavior {behavior_id}",
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("Command Embeddings - UMAP 2D Projection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_results(output_dir, embeddings, projection_2d, behaviors, text_commands, metrics):
    """Save all results to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(str(Path(output_dir) / "embeddings.npy"), embeddings)
    np.save(str(Path(output_dir) / "umap_projection_2d.npy"), projection_2d)
    np.save(str(Path(output_dir) / "behaviors.npy"), behaviors)

    with open(Path(output_dir) / "text_commands.txt", "w") as f:
        for cmd in text_commands:
            f.write(f"{cmd}\n")

    with open(Path(output_dir) / "cluster_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ===================== MAIN =====================

if __name__ == "__main__":
    # ========== HYPERPARAMETERS ==========
    # JSON with your commands:
    # located at: root_folder / "dataset" / "master_files.json"
    text_commands_path = "dataset/master_file_gen_me.json"

    output_base_path = "dataset_generation"
    k_per_behavior = 100  # will sample up to this many per behavior

    encoder_mode = "FROZEN"  # "TRAINED" or "FROZEN"
    checkpoint_path = "freeflyer/decision_transformer/saved_files/checkpoints/v_03/text_adapter.pth"

    model_name = "distilbert-base-uncased"
    out_dim = 384
    max_tokens = 50
    batch_size = 32
    random_seed = 42

    device = "auto"  # "auto", "cuda", or "cpu"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    umap_n_neighbors = 30
    umap_min_dist = 1.0
    umap_n_components = 2
    # =====================================

    # Load data from JSON
    print("Loading data from JSON...")
    text_commands, behaviors = load_data_from_json(text_commands_path, root_folder)

    # Sample per behavior
    print(f"Sampling up to {k_per_behavior} commands per behavior...")
    sampled_texts, sampled_behaviors = sample_by_behavior(
        text_commands, behaviors, k_per_behavior, random_seed
    )

    # Create encoder
    print("Creating text encoder...")
    encoder = create_text_encoder(
        encoder_mode,
        checkpoint_path,
        model_name,
        out_dim,
        max_tokens,
        device,
        root_folder,
    )

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(sampled_texts, encoder, batch_size, device)
    print(f"Embeddings shape: {embeddings.shape}")

    # Apply UMAP
    print("Applying UMAP...")
    projection_2d = apply_umap(
        embeddings, umap_n_neighbors, umap_min_dist, umap_n_components, random_seed
    )

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(embeddings, sampled_behaviors, projection_2d)

    # Visualize
    print("Creating visualization...")
    output_dir = str(root_folder / output_base_path / "UMAP_Results")
    os.makedirs(output_dir, exist_ok=True)
    visualize(
        projection_2d,
        sampled_behaviors,
        str(Path(output_dir) / "visualization_2d.png"),
    )

    # Save results
    print("Saving results...")
    save_results(
        output_dir, embeddings, projection_2d, sampled_behaviors, sampled_texts, metrics
    )

    print(f"\nDone! Results saved to: {output_dir}")
    print(f"Overall silhouette score: {metrics['silhouette_overall']:.4f}")
    print(f"Nearest-neighbor accuracy: {metrics['nearest_neighbor_accuracy']:.4f}")
