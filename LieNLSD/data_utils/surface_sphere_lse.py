import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from data_utils.surface_sphere import make_query_points, sample_sphere_surface


def load_lse_class(symmetryml_root):
    src_path = Path(symmetryml_root).expanduser().resolve() / "src"
    if not src_path.exists():
        raise FileNotFoundError(
            f"Could not find SymmetryML src directory at {src_path}. "
            "Pass --symmetryml_root pointing to the cloned SymmetryML repo."
        )
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    try:
        from symdisc.discovery.lse.core import LSE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import SymmetryML LSE dependencies. "
            "Make sure SymmetryML requirements are installed, especially scikit-learn."
        ) from exc
    return LSE


def fit_lse(surface_points, symmetryml_root, degree, rel_tol, seed):
    LSE = load_lse_class(symmetryml_root)
    lse = LSE(
        mode="polynomial",
        degree=degree,
        include_bias=False,
        use_incremental=False,
        lowvar_policy="relative",
        rel_tol=rel_tol,
        n_components=None,
        svd_solver="randomized",
        random_state=seed,
    ).fit(surface_points)
    return lse


def evaluate_lse_target(lse, points):
    constraints = lse.constraint_values(points)
    target = np.linalg.norm(constraints, axis=1)
    return target.astype(np.float32), constraints.astype(np.float32)


def save_split(save_dir, split, surface_points, query_points, target, occupancy, constraints):
    split_dir = Path(save_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(surface_points).to(torch.float32), split_dir / f"{split}-surface.pt")
    torch.save(torch.from_numpy(query_points).to(torch.float32), split_dir / f"{split}-query.pt")
    torch.save(torch.from_numpy(target).to(torch.float32), split_dir / f"{split}-target.pt")
    torch.save(torch.from_numpy(occupancy).to(torch.float32), split_dir / f"{split}-occupancy.pt")
    torch.save(torch.from_numpy(constraints).to(torch.float32), split_dir / f"{split}-constraints.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_surface", type=int, default=4000)
    parser.add_argument("--n_query", type=int, default=4000)
    parser.add_argument("--query_sigma", type=float, default=0.05)
    parser.add_argument("--save_dir", type=str, default="./data/surface_sphere_lse")
    parser.add_argument("--save_name", type=str, default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--symmetryml_root",
        type=str,
        default="../SymmetryML",
        help="Path to the cloned SymmetryML repository.",
    )
    parser.add_argument("--lse_degree", type=int, default=3)
    parser.add_argument("--lse_rel_tol", type=float, default=1e-8)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    surface_points = sample_sphere_surface(args.n_surface, rng)
    lse = fit_lse(
        surface_points=surface_points,
        symmetryml_root=args.symmetryml_root,
        degree=args.lse_degree,
        rel_tol=args.lse_rel_tol,
        seed=args.seed,
    )
    query_points, _, occupancy = make_query_points(
        surface_points=surface_points,
        n_query=args.n_query,
        sigma=args.query_sigma,
        rng=rng,
    )
    target, constraints = evaluate_lse_target(lse, query_points)
    save_split(
        save_dir=args.save_dir,
        split=args.save_name,
        surface_points=surface_points,
        query_points=query_points,
        target=target,
        occupancy=occupancy,
        constraints=constraints,
    )
    print(
        f"Saved {args.save_name} split with {surface_points.shape[0]} surface points, "
        f"{query_points.shape[0]} query points, and {constraints.shape[1]} LSE constraint(s)."
    )
