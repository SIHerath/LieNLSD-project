import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import get_dataset
from discovery import symmetry_discovery
from model import MLP


def sphere_truth_generators_linear():
    rx = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    ry = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ])
    rz = np.array([
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    return np.stack([rx, ry, rz], axis=0)


def sphere_truth_generators_quadratic():
    g = np.zeros((3, 3, 10), dtype=float)

    # Rx = y ∂z - z ∂y
    g[0, 1, 3] = -1.0
    g[0, 2, 2] = 1.0

    # Ry = z ∂x - x ∂z
    g[1, 0, 3] = 1.0
    g[1, 2, 1] = -1.0

    # Rz = x ∂y - y ∂x
    g[2, 0, 2] = -1.0
    g[2, 1, 1] = 1.0

    return g


def flatten_generators(W):
    return W.reshape(W.shape[0], -1).T


def orthonormal_basis(G, tol=1e-10):
    U, S, _ = np.linalg.svd(G, full_matrices=False)
    rank = int(np.sum(S > tol))
    return U[:, :rank], S[:rank]


def principal_angle_cosines(G_true, G_pred):
    Q_true, S_true = orthonormal_basis(G_true)
    Q_pred, S_pred = orthonormal_basis(G_pred)
    M = Q_true.T @ Q_pred
    s = np.linalg.svd(M, compute_uv=False)
    return s, Q_true.shape[1], Q_pred.shape[1], S_true, S_pred


def overlap_precision_recall_f1(s, true_dim, pred_dim):
    overlap = float(np.sum(s**2))
    recall = overlap / true_dim if true_dim > 0 else 0.0
    precision = overlap / pred_dim if pred_dim > 0 else 0.0
    f1 = 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return overlap, recall, precision, f1


def grassmann_geodesic_distance(s):
    s = np.clip(s, -1.0, 1.0)
    theta = np.arccos(s)
    distance = float(np.linalg.norm(theta))
    return theta, distance


def get_basis_dim(basis):
    if basis == "linear":
        return 4
    if basis == "quadratic":
        return 10
    raise ValueError(f"Unknown basis: {basis}")


def get_truth_generators(basis):
    if basis == "linear":
        return sphere_truth_generators_linear()
    if basis == "quadratic":
        return sphere_truth_generators_quadratic()
    raise ValueError(f"Unknown basis: {basis}")


def load_saved_generators(path, basis):
    basis_dim = get_basis_dim(basis)
    x = torch.load(path, weights_only=True)
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    x = np.asarray(x, dtype=float)

    if x.ndim == 3:
        return x

    flat_dim = 3 * basis_dim
    if x.ndim == 2:
        if x.shape[1] == flat_dim:
            return x.reshape(x.shape[0], 3, basis_dim)
        if x.shape[0] == flat_dim:
            return x.T.reshape(x.shape[1], 3, basis_dim)

    raise ValueError(
        f"Could not interpret generator file with shape {x.shape}. "
        "Expected (n,3,basis_dim) or a 2D flattened generator matrix."
    )


def run_raw_discovery(task, save_dir, epoch, threshold, basis, sample, num_discovery_samples, device):
    args = {
        "task": task,
        "hidden_dim": 200,
        "n_layers": 3,
        "activation": "Sigmoid",
        "save_dir": save_dir,
        "epoch": epoch,
        "device": torch.device(device),
        "gpu": 0,
        "log_interval": 1000,
        "threshold": threshold,
        "seed": 0,
        "sample": sample,
        "epsilon1": 1e-4,
        "epsilon2": 1e-4,
        "sphere_basis": basis,
        "num_discovery_samples": num_discovery_samples,
    }

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    train_dataset, _, args = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model = MLP(**args).to(args["device"])
    model.load_state_dict(
        torch.load(
            f"saved_models/{save_dir}/model_{epoch}.pt",
            weights_only=True,
            map_location=device,
        )
    )
    _, Vh = symmetry_discovery(model=model, train_loader=train_loader, **args)
    basis_dim = get_basis_dim(basis)
    return Vh.cpu().numpy().reshape(Vh.shape[0], 3, basis_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--generator_path", type=str, default=None,
                        help="Path to a saved generator.pt or flattened generator tensor.")
    parser.add_argument("--raw_task", type=str, default=None,
                        help="If set, run raw symmetry discovery instead of loading generator_path.")
    parser.add_argument("--save_dir", type=str, default="sphere")
    parser.add_argument("--epoch", type=int, default=199)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--num_discovery_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.generator_path is None and args.raw_task is None:
        raise ValueError("Provide either --generator_path or --raw_task.")

    if args.generator_path is not None and args.raw_task is not None:
        raise ValueError("Use either --generator_path or --raw_task, not both.")

    if args.raw_task is not None:
        W_pred = run_raw_discovery(
            task=args.raw_task,
            save_dir=args.save_dir,
            epoch=args.epoch,
            threshold=args.threshold,
            basis=args.basis,
            sample=args.sample,
            num_discovery_samples=args.num_discovery_samples,
            device=args.device,
        )
        source = f"raw discovery task={args.raw_task}"
    else:
        W_pred = load_saved_generators(args.generator_path, args.basis)
        source = args.generator_path

    W_true = get_truth_generators(args.basis)

    G_pred = flatten_generators(W_pred)
    G_true = flatten_generators(W_true)

    s, true_dim, pred_dim, _, _ = principal_angle_cosines(G_true, G_pred)
    overlap, recall, precision, f1 = overlap_precision_recall_f1(s, true_dim, pred_dim)
    theta, grassmann_distance = grassmann_geodesic_distance(s)

    print(f"Source: {source}")
    print(f"Basis: {args.basis}")
    print(f"Predicted generator tensor shape: {tuple(W_pred.shape)}")
    print(f"True subspace dimension: {true_dim}")
    print(f"Predicted subspace dimension: {pred_dim}")
    print("Principal-angle cosines:", " ".join(f"{v:.8f}" for v in s.tolist()))
    print("Principal angles (radians):", " ".join(f"{v:.8f}" for v in theta.tolist()))
    print("Principal angles (degrees):", " ".join(f"{v:.4f}" for v in np.degrees(theta).tolist()))
    print(f"Grassmann geodesic distance: {grassmann_distance:.8f}")
    print(f"Subspace score (mean cosine^2): {float(np.mean(s**2)):.8f}")
    print(f"Overlap energy sum(cos^2): {overlap:.8f}")
    print(f"Recall (normalized by true dim): {recall:.8f}")
    print(f"Precision (normalized by predicted dim): {precision:.8f}")
    print(f"F1 score: {f1:.8f}")
