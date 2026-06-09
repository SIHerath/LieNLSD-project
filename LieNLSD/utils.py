import math
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def basis_sparsification(Q, epsilon1, epsilon2, device, log_interval, **kwargs):
    max_iter = kwargs.get('max_iter', 20000)
    return_device = device
    work_device = torch.device('cpu')
    
    def soft_threshold(x, epsilon):
        return torch.sign(x) * torch.clamp(torch.abs(x) - epsilon, min=0.0)

    # Run sparsification on CPU because MPS does not implement the spectral
    # norm / SVD path used here. The matrix is small, so the overhead is minor.
    Q = Q.detach().to(work_device, dtype=torch.float32)
    n, d = Q.shape
    beta = min(n, d)
    beta_max = 1e10
    rho0 = 1.9
    etaA = 1.02 * torch.linalg.norm(Q, ord=2) ** 2
    etaB = 1.02
    R = torch.eye(d, device=work_device)
    Z = Q @ R
    Lambda = torch.zeros_like(Q)
    k = 0
    while True:
        R_pre = R.clone()
        Z_pre = Z.clone()
        tempR = R - (Q.T @ (Lambda + beta * (Q @ R - Z))) / (beta * etaA)
        tempR = Q.T @ Z / etaA - Q.T @ Lambda / (beta * etaA)
        # MPS does not support float64 SVD, so do this decomposition on CPU.
        tempR_cpu64 = tempR.detach().to('cpu').to(torch.float64)
        U, S, Vh = torch.linalg.svd(tempR_cpu64)
        U = U.to(device=work_device, dtype=torch.float32)
        Vh = Vh.to(device=work_device, dtype=torch.float32)
        R = U @ Vh
        tempZ = Z + (Lambda + beta * (Q @ R - Z)) / (beta * etaB)
        Z = soft_threshold(tempZ, 1.0 / (beta * etaB))
        Lambda = Lambda + beta * (Q @ R - Z)
        condition = beta * max(math.sqrt(etaA) * torch.linalg.norm(R - R_pre, ord=float('inf')).item(), math.sqrt(etaB) * torch.linalg.norm(Z - Z_pre, ord=float('inf')).item())
        if condition < epsilon2:
            rho = rho0
        else:
            rho = 1.0
        beta = min(beta_max, rho * beta)
        loss = torch.sum(torch.abs(Q @ R)).item()
        if (k + 1) % log_interval == 0:
            print(f'Epoch {k}, loss: {loss:.4f}')
        k += 1
        if torch.linalg.norm(Q @ R - Z, ord=float('inf')) < epsilon1 and condition <= epsilon2:
            return (Q @ R).to(return_device)
        if k >= max_iter:
            raise RuntimeError(
                f'basis_sparsification did not converge within {max_iter} iterations '
                f'(epsilon1={epsilon1}, epsilon2={epsilon2}, last_loss={loss:.4f}, '
                f'last_condition={condition:.4e})'
            )

save_vis_path = './vis'


def sphere_basis_labels(sphere_basis):
    if sphere_basis == 'linear':
        return [r'$1$', r'$x$', r'$y$', r'$z$']
    if sphere_basis == 'quadratic':
        return [r'$1$', r'$x$', r'$y$', r'$z$', r'$x^2$', r'$y^2$', r'$z^2$', r'$xy$', r'$xz$', r'$yz$']
    raise ValueError(f'Unknown sphere basis: {sphere_basis}')


def vis(S, Vh, task, sample, sphere_basis='quadratic'):
    S = S.cpu().numpy()
    n = Vh.shape[0]
    start = max(0, S.shape[0] - max(n, 1) - 1)
    x_vals = np.arange(start + 1, S.shape[0] + 1)
    y_vals = S[start:]
    plt.plot(x_vals, y_vals)
    plt.xticks(x_vals)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Singular Value')
    if not os.path.exists(f'{save_vis_path}/{task}_{sample}'):
        os.makedirs(f'{save_vis_path}/{task}_{sample}')
    plt.savefig(f'{save_vis_path}/{task}_{sample}/singular_value.png', bbox_inches='tight')
    plt.clf()

    if task == 'top':
        W = Vh.reshape(n, 4, 4)
    elif task in ['heat', 'burger', 'kdv']:
        W = Vh.reshape(n, 3, 10)
    elif task in ['sphere', 'sphere_truth', 'sphere_lse', 'cylinder']:
        sphere_dim = 4 if sphere_basis == 'linear' else 10
        W = Vh.reshape(n, 3, sphere_dim)
    elif task == 'wave':
        W = Vh.reshape(n, 4, 15)
    elif task in ['rd', 'schrodinger']:
        W = Vh.reshape(n, 5, 21)
    torch.save(W, f'{save_vis_path}/{task}_{sample}/generator.pt')
    W = W.cpu().numpy()
    for i in range(n):
        plt.imshow(W[i])
        if task == 'top':
            plt.colorbar()
            plt.xticks(ticks=np.arange(4), labels=[r'$p^0$', r'$p^1$', r'$p^2$', r'$p^3$'])
            plt.yticks(ticks=np.arange(4), labels=[r'$\partial_{p^0}$', r'$\partial_{p^1}$', r'$\partial_{p^2}$', r'$\partial_{p^3}$'])
        elif task in ['heat', 'burger', 'kdv']:
            plt.colorbar(shrink=0.5)
            plt.xticks(ticks=np.arange(10), labels=[r'$1$', r'$t$', r'$x$', r'$u$', r'$t^2$', r'$x^2$', r'$u^2$', r'$tx$', r'$tu$', r'$xu$'])
            plt.yticks(ticks=np.arange(3), labels=[r'$\partial_t$', r'$\partial_x$', r'$\partial_u$'])
        elif task in ['sphere', 'sphere_truth', 'sphere_lse', 'cylinder']:
            plt.colorbar(shrink=0.5)
            labels = sphere_basis_labels(sphere_basis)
            plt.xticks(ticks=np.arange(len(labels)), labels=labels)
            plt.yticks(ticks=np.arange(3), labels=[r'$\partial_x$', r'$\partial_y$', r'$\partial_z$'])
        elif task == 'wave':
            plt.colorbar(shrink=0.5)
            plt.xticks(ticks=np.arange(15), labels=[r'$1$', r'$t$', r'$x$', r'$y$', r'$u$', r'$t^2$', r'$x^2$', r'$y^2$', r'$u^2$', r'$tx$', r'$ty$', r'$tu$', r'$xy$', r'$xu$', r'$yu$'])
            plt.yticks(ticks=np.arange(4), labels=[r'$\partial_t$', r'$\partial_x$', r'$\partial_y$', r'$\partial_u$'])
        elif task in ['rd', 'schrodinger']:
            plt.colorbar(shrink=0.5)
            plt.xticks(ticks=np.arange(21), labels=[r'$1$', r'$t$', r'$x$', r'$y$', r'$u$', r'$v$', r'$t^2$', r'$x^2$', r'$y^2$', r'$u^2$', r'$v^2$', r'$tx$', r'$ty$', r'$tu$', r'$tv$', r'$xy$', r'$xu$', r'$xv$', r'$yu$', r'$yv$', r'$uv$'])
            plt.yticks(ticks=np.arange(5), labels=[r'$\partial_t$', r'$\partial_x$', r'$\partial_y$', r'$\partial_u$', r'$\partial_v$'])
        plt.tick_params(axis='both', which='both', length=0)
        plt.savefig(f'{save_vis_path}/{task}_{sample}/generator_{S.shape[0] - n + i + 1}.png', bbox_inches='tight')
        plt.clf()
