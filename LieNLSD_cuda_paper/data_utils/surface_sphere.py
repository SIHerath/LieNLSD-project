import argparse
import math
from pathlib import Path

import numpy as np
import torch


def surface_area_density(f, g, u_grid):
    f_vals = f(u_grid)
    g_vals = g(u_grid)
    f_prime = np.gradient(f_vals, u_grid)
    g_prime = np.gradient(g_vals, u_grid)
    return np.abs(f_vals) * np.sqrt(f_prime**2 + g_prime**2)


def sample_surface_of_revolution_rejection(f, g, u_min, u_max, n, rng, grid_size=2000):
    u_grid = np.linspace(u_min, u_max, grid_size)
    w_grid = surface_area_density(f, g, u_grid)
    w_max = np.max(w_grid)
    if w_max <= 0:
        raise ValueError('Surface area density is non-positive everywhere.')

    accepted = []
    while len(accepted) < n:
        batch_size = max(1024, 2 * (n - len(accepted)))
        u_prop = rng.uniform(u_min, u_max, size=batch_size)
        accept_prob = np.interp(u_prop, u_grid, w_grid) / w_max
        keep = rng.uniform(0.0, 1.0, size=batch_size) < accept_prob
        accepted.extend(u_prop[keep].tolist())

    u = np.asarray(accepted[:n], dtype=np.float64)
    v = rng.uniform(0.0, 2.0 * math.pi, size=n)

    radius = f(u)
    z = g(u)
    x = radius * np.cos(v)
    y = radius * np.sin(v)
    return np.stack([x, y, z], axis=1)


def sample_sphere_surface(n_surface, rng):
    f = lambda u: np.sin(u)
    g = lambda u: np.cos(u)
    return sample_surface_of_revolution_rejection(
        f=f,
        g=g,
        u_min=0.0,
        u_max=math.pi,
        n=n_surface,
        rng=rng,
    )


def make_query_points(surface_points, n_query, sigma, rng):
    idx = rng.integers(0, surface_points.shape[0], size=n_query)
    base = surface_points[idx]
    radii = np.linalg.norm(base, axis=1, keepdims=True)
    normals = base / np.clip(radii, 1e-12, None)
    offsets = rng.normal(loc=0.0, scale=sigma, size=(n_query, 1))
    query = base + offsets * normals
    sdf = np.linalg.norm(query, axis=1) - 1.0
    occupancy = (sdf <= 0.0).astype(np.float32)
    return query, sdf.astype(np.float32), occupancy


def save_split(save_dir, split, surface_points, query_points, sdf, occupancy):
    split_dir = Path(save_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(surface_points).to(torch.float32), split_dir / f'{split}-surface.pt')
    torch.save(torch.from_numpy(query_points).to(torch.float32), split_dir / f'{split}-query.pt')
    torch.save(torch.from_numpy(sdf).to(torch.float32), split_dir / f'{split}-sdf.pt')
    torch.save(torch.from_numpy(occupancy).to(torch.float32), split_dir / f'{split}-occupancy.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_surface', type=int, default=4000)
    parser.add_argument('--n_query', type=int, default=4000)
    parser.add_argument('--query_sigma', type=float, default=0.05)
    parser.add_argument('--save_dir', type=str, default='./data/surface_sphere')
    parser.add_argument('--save_name', type=str, default='train')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    surface_points = sample_sphere_surface(args.n_surface, rng)
    query_points, sdf, occupancy = make_query_points(
        surface_points=surface_points,
        n_query=args.n_query,
        sigma=args.query_sigma,
        rng=rng,
    )
    save_split(
        save_dir=args.save_dir,
        split=args.save_name,
        surface_points=surface_points,
        query_points=query_points,
        sdf=sdf,
        occupancy=occupancy,
    )
