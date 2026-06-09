import argparse
import csv
import json
import os
import time
import traceback
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from adan import Adan
from compare_sphere_subspaces import (
    flatten_generators,
    get_basis_dim,
    get_truth_generators,
    grassmann_geodesic_distance,
    overlap_precision_recall_f1,
    principal_angle_cosines,
)
from dataset import get_dataset
from discovery import symmetry_discovery
from model import MLP
from parser_utils import resolve_device
from utils import basis_sparsification


CSV_FIELDS = [
    "method",
    "task",
    "mode",
    "activation",
    "basis",
    "threshold",
    "num_train",
    "seed",
    "device",
    "epochs",
    "train_time_sec",
    "raw_discovery_time_sec",
    "sparsification_time_sec",
    "total_time_sec",
    "train_loss",
    "val_loss",
    "raw_dim",
    "raw_cosines",
    "raw_angles_rad",
    "raw_grassmann",
    "raw_recall",
    "raw_precision",
    "raw_f1",
    "final_dim",
    "final_cosines",
    "final_angles_rad",
    "final_grassmann",
    "final_recall",
    "final_precision",
    "final_f1",
    "sparsification_status",
    "error_message",
    "run_dir",
]


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_array(values):
    if values is None:
        return ""
    return " ".join(f"{float(v):.8f}" for v in values)


def label_from_batch(data, task, device):
    feature = torch.stack([data["x"], data["y"], data["z"]], dim=1).to(device)
    if task == "sphere":
        label = data["sdf"].unsqueeze(1).to(device)
    elif task in ["sphere_lse", "sphere_truth"]:
        label = data["target"].unsqueeze(1).to(device)
    else:
        raise ValueError(f"Unsupported benchmark task: {task}")
    return feature, label


def evaluate_loss(model, loader, task, device):
    loss_fn = torch.nn.MSELoss()
    losses = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            feature, label = label_from_batch(data, task, device)
            losses.append(loss_fn(model(feature), label).item())
    return float(np.mean(losses)) if losses else float("nan")


def train_model_for_run(model, train_loader, val_loader, args, device):
    if args["opt"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), args["lr"])
    elif args["opt"] == "Adan":
        optimizer = Adan(model.parameters(), args["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {args['opt']}")

    loss_fn = torch.nn.MSELoss()
    for _ in range(args["epochs"]):
        model.train()
        for data in train_loader:
            feature, label = label_from_batch(data, args["task"], device)
            loss = loss_fn(model(feature), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return evaluate_loss(model, train_loader, args["task"], device), evaluate_loss(model, val_loader, args["task"], device)


def make_subset(dataset, num_train, seed):
    if num_train > len(dataset):
        raise ValueError(
            f"Requested num_train={num_train}, but dataset only has {len(dataset)} examples. "
            "Regenerate the dataset with more query points before running this sample size."
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_train].tolist()
    return Subset(dataset, indices), indices


def compute_similarity(generator_tensor, basis):
    if generator_tensor is None or generator_tensor.shape[0] == 0:
        return {
            "dim": 0,
            "cosines": [],
            "angles": [],
            "grassmann": float("nan"),
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }

    pred = generator_tensor.detach().cpu().numpy()
    truth = get_truth_generators(basis)
    cosines, true_dim, pred_dim, _, _ = principal_angle_cosines(
        flatten_generators(truth),
        flatten_generators(pred),
    )
    _, recall, precision, f1 = overlap_precision_recall_f1(cosines, true_dim, pred_dim)
    angles, grassmann = grassmann_geodesic_distance(cosines)
    return {
        "dim": int(pred_dim),
        "cosines": cosines,
        "angles": angles,
        "grassmann": grassmann,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def run_one(method, config, num_train, seed, device, run_root):
    run_name = f"{method['name']}_N{num_train}_seed{seed}"
    run_dir = os.path.join(run_root, method["name"], f"N_{num_train}", f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    args = {
        "task": method["task"],
        "opt": config["opt"],
        "batch_size": config["batch_size"],
        "num_epochs": config["epochs"],
        "epochs": config["epochs"],
        "lr": config["lr"],
        "hidden_dim": config["hidden_dim"],
        "n_layers": config["n_layers"],
        "activation": method["activation"],
        "device": device,
        "gpu": 0,
        "sphere_basis": method["basis"],
        "log_interval": 10**9,
        "save_interval": 10**9,
        "save_dir": run_name,
        "seed": seed,
        "threshold": method["threshold"],
        "sample": config["discovery_sample"],
        "num_discovery_samples": config["num_discovery_samples"],
        "epsilon1": config["epsilon1"],
        "epsilon2": config["epsilon2"],
        "max_iter": config["max_sparsification_iter"],
    }
    if config.get("surface_data_dir"):
        args["surface_data_dir"] = config["surface_data_dir"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_dataset, val_dataset, args = get_dataset(args)
    train_subset, subset_indices = make_subset(train_dataset, num_train, seed)
    torch.save(torch.tensor(subset_indices), os.path.join(run_dir, "train_indices.pt"))

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    discovery_loader = DataLoader(train_subset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(seed))

    model = MLP(**args).to(device)

    total_start = time.perf_counter()

    sync_if_needed(device)
    train_start = time.perf_counter()
    train_loss, val_loss = train_model_for_run(model, train_loader, val_loader, args, device)
    sync_if_needed(device)
    train_time = time.perf_counter() - train_start

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    sync_if_needed(device)
    raw_start = time.perf_counter()
    singular_values, raw_vh = symmetry_discovery(model=model, train_loader=discovery_loader, **args)
    sync_if_needed(device)
    raw_time = time.perf_counter() - raw_start

    basis_dim = get_basis_dim(method["basis"])
    raw_generators = raw_vh.reshape(raw_vh.shape[0], 3, basis_dim).detach().cpu()
    torch.save(singular_values.detach().cpu(), os.path.join(run_dir, "singular_values.pt"))
    torch.save(raw_generators, os.path.join(run_dir, "raw_generators.pt"))
    raw_metrics = compute_similarity(raw_generators, method["basis"])

    final_generators = None
    final_metrics = None
    sparsification_status = "not_run"
    error_message = ""
    sync_if_needed(device)
    sparse_start = time.perf_counter()
    if config.get("skip_sparsification", False):
        sparsification_status = "skipped"
    else:
        try:
            sparse_vh = basis_sparsification(Q=raw_vh.T, **args).T
            sync_if_needed(device)
            final_generators = sparse_vh.reshape(sparse_vh.shape[0], 3, basis_dim).detach().cpu()
            torch.save(final_generators, os.path.join(run_dir, "sparse_generators.pt"))
            final_metrics = compute_similarity(final_generators, method["basis"])
            sparsification_status = "success"
        except Exception as exc:
            sync_if_needed(device)
            sparsification_status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            with open(os.path.join(run_dir, "sparsification_error.txt"), "w", encoding="utf-8") as f:
                f.write(error_message)
                f.write("\n\n")
                f.write(traceback.format_exc())
    sparse_time = time.perf_counter() - sparse_start
    total_time = time.perf_counter() - total_start

    if final_metrics is None:
        final_metrics = {
            "dim": "",
            "cosines": None,
            "angles": None,
            "grassmann": "",
            "recall": "",
            "precision": "",
            "f1": "",
        }

    row = {
        "method": method["name"],
        "task": method["task"],
        "mode": method["mode"],
        "activation": method["activation"],
        "basis": method["basis"],
        "threshold": method["threshold"],
        "num_train": num_train,
        "seed": seed,
        "device": str(device),
        "epochs": config["epochs"],
        "train_time_sec": f"{train_time:.6f}",
        "raw_discovery_time_sec": f"{raw_time:.6f}",
        "sparsification_time_sec": f"{sparse_time:.6f}",
        "total_time_sec": f"{total_time:.6f}",
        "train_loss": f"{train_loss:.10g}",
        "val_loss": f"{val_loss:.10g}",
        "raw_dim": raw_metrics["dim"],
        "raw_cosines": format_array(raw_metrics["cosines"]),
        "raw_angles_rad": format_array(raw_metrics["angles"]),
        "raw_grassmann": f"{raw_metrics['grassmann']:.8f}",
        "raw_recall": f"{raw_metrics['recall']:.8f}",
        "raw_precision": f"{raw_metrics['precision']:.8f}",
        "raw_f1": f"{raw_metrics['f1']:.8f}",
        "final_dim": final_metrics["dim"],
        "final_cosines": format_array(final_metrics["cosines"]),
        "final_angles_rad": format_array(final_metrics["angles"]),
        "final_grassmann": final_metrics["grassmann"] if final_metrics["grassmann"] == "" else f"{final_metrics['grassmann']:.8f}",
        "final_recall": final_metrics["recall"] if final_metrics["recall"] == "" else f"{final_metrics['recall']:.8f}",
        "final_precision": final_metrics["precision"] if final_metrics["precision"] == "" else f"{final_metrics['precision']:.8f}",
        "final_f1": final_metrics["f1"] if final_metrics["f1"] == "" else f"{final_metrics['f1']:.8f}",
        "sparsification_status": sparsification_status,
        "error_message": error_message,
        "run_dir": run_dir,
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    return row


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sphere_benchmark.json")
    parser.add_argument("--output", default="results/sphere_benchmark.csv")
    parser.add_argument("--run_root", default="benchmark_runs")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample_sizes", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=None, help="Optional list of method names to run.")
    parser.add_argument("--max_sparsification_iter", type=int, default=None)
    parser.add_argument("--skip_sparsification", action="store_true")
    parser.add_argument("--surface_data_dir", default=None)
    return parser.parse_args()


def main():
    cli = parse_args()
    config = load_config(cli.config)

    if cli.sample_sizes is not None and len(cli.sample_sizes) > 0:
        config["sample_sizes"] = cli.sample_sizes
    if cli.seeds is not None and len(cli.seeds) > 0:
        config["seeds"] = cli.seeds
    if cli.epochs is not None:
        config["epochs"] = cli.epochs
    if cli.max_sparsification_iter is not None:
        config["max_sparsification_iter"] = cli.max_sparsification_iter
    if cli.skip_sparsification:
        config["skip_sparsification"] = True
    if cli.surface_data_dir is not None:
        config["surface_data_dir"] = cli.surface_data_dir

    if cli.methods:
        selected = set(cli.methods)
        config["methods"] = [m for m in config["methods"] if m["name"] in selected]
        missing = selected - {m["name"] for m in config["methods"]}
        if missing:
            raise ValueError(f"Unknown method names: {sorted(missing)}")

    device = resolve_device(cli.device, cli.gpu)
    os.makedirs(os.path.dirname(cli.output) or ".", exist_ok=True)
    os.makedirs(cli.run_root, exist_ok=True)

    rows = []
    output_exists = os.path.exists(cli.output)
    with open(cli.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not output_exists:
            writer.writeheader()

        for method in config["methods"]:
            for num_train in config["sample_sizes"]:
                for seed in config["seeds"]:
                    print(f"Running {method['name']} N={num_train} seed={seed} on {device}")
                    try:
                        row = run_one(deepcopy(method), config, num_train, seed, device, cli.run_root)
                    except Exception as exc:
                        row = {
                            field: "" for field in CSV_FIELDS
                        }
                        row.update({
                            "method": method["name"],
                            "task": method["task"],
                            "mode": method["mode"],
                            "activation": method["activation"],
                            "basis": method["basis"],
                            "threshold": method["threshold"],
                            "num_train": num_train,
                            "seed": seed,
                            "device": str(device),
                            "epochs": config["epochs"],
                            "sparsification_status": "run_failed",
                            "error_message": f"{type(exc).__name__}: {exc}",
                        })
                        print(row["error_message"])
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)

    print(f"Wrote {len(rows)} rows to {cli.output}")


if __name__ == "__main__":
    main()
