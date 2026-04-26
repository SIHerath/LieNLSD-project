import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import get_dataset
from discovery import symmetry_discovery
from model import MLP


BASIS = ['1', 'x', 'y', 'z', 'x^2', 'y^2', 'z^2', 'xy', 'xz', 'yz']
DERS = ['∂x', '∂y', '∂z']


def poly_str(coeffs, thresh):
    terms = []
    for coeff, basis_term in zip(coeffs, BASIS):
        coeff = float(coeff)
        if abs(coeff) < thresh:
            continue
        magnitude = abs(coeff)
        if basis_term == '1':
            body = f'{magnitude:.4f}'
        else:
            body = basis_term if abs(magnitude - 1.0) < 5e-3 else f'{magnitude:.4f} {basis_term}'
        if not terms:
            terms.append(('-' if coeff < 0 else '') + body)
        else:
            terms.append((' - ' if coeff < 0 else ' + ') + body)
    return ''.join(terms) if terms else '0'


def generator_str(vec, coeff_thresh):
    W = vec.reshape(3, 10).numpy()
    parts = []
    for row, derivative in zip(W, DERS):
        poly = poly_str(row, coeff_thresh)
        if poly != '0':
            parts.append(f'({poly}) {derivative}')
    return ' + '.join(parts) if parts else '0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='sphere')
    parser.add_argument('--epoch', type=int, default=199)
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--coeff_threshold', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--activation', type=str, default='Sigmoid')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_args = {
        'task': 'sphere',
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'activation': args.activation,
        'save_dir': args.save_dir,
        'epoch': args.epoch,
        'device': torch.device('cpu'),
        'gpu': 0,
        'log_interval': 1000,
        'threshold': args.threshold,
        'seed': args.seed,
        'sample': args.sample,
        'epsilon1': 1e-4,
        'epsilon2': 1e-4,
    }

    train_dataset, _, run_args = get_dataset(run_args)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = MLP(**run_args).to(run_args['device'])
    checkpoint_path = f'saved_models/{args.save_dir}/model_{args.epoch}.pt'
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True, map_location=run_args['device'])
    )

    S, Vh = symmetry_discovery(model=model, train_loader=train_loader, **run_args)

    print('SINGULAR_VALUES')
    print(' '.join(f'{float(s):.8g}' for s in S.tolist()))
    print('NUM_SELECTED', Vh.shape[0])

    for i, vec in enumerate(Vh, 1):
        print(f'RAW_{i}_GENERATOR')
        print(generator_str(vec, args.coeff_threshold))


if __name__ == '__main__':
    main()
