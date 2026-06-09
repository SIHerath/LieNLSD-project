import argparse
import torch

def resolve_device(device_name, gpu_index):
    if device_name == 'auto':
        if torch.cuda.is_available() and gpu_index != -1:
            return torch.device(f'cuda:{gpu_index}')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available.')
        if gpu_index == -1:
            raise RuntimeError('CUDA was requested with --gpu -1.')
        return torch.device(f'cuda:{gpu_index}')
    if device_name == 'mps':
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError('MPS was requested but is not available.')
        return torch.device('mps')
    if device_name == 'cpu':
        return torch.device('cpu')
    raise ValueError(f'Unknown device option: {device_name}')

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='heat')

    parser.add_argument('--opt', type=str, default='Adan')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--activation', type=str, default='ReLU')

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sphere_basis', type=str, default='quadratic', choices=['linear', 'quadratic'])
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='test')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    args.device = resolve_device(args.device, args.gpu)

    return args

def get_discovery_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='heat')

    parser.add_argument('--epsilon1', type=float, default=1e-4)
    parser.add_argument('--epsilon2', type=float, default=1e-5)

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--save_dir', type=str, default='test')
    parser.add_argument('--epoch', type=int, default=19)

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sphere_basis', type=str, default='quadratic', choices=['linear', 'quadratic'])
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--num_discovery_samples', type=int, default=100,
                        help='Number of points used in discovery. Use -1 to use all points in the training split.')

    args = parser.parse_args()

    args.device = resolve_device(args.device, args.gpu)

    return args
