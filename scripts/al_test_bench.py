#!/usr/bin/env python3
"""
Active Learning Test Bench
==========================

Fast testing of acquisition functions using SVGP surrogate.
Reports where the model samples across iterations.

Usage:
    python scripts/al_test_bench.py --data data/sobol_4q/train.pt --n_pool 50000
"""

import argparse
import time
import numpy as np
import torch
import gpytorch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import MiniBatchKMeans


class FastSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class FastSurrogate:
    def __init__(self, num_inducing=150, num_epochs=20, lr=0.05):
        self.num_inducing = num_inducing
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.likelihood = None

    def fit(self, X, y):
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32).squeeze()

        n_inducing = min(self.num_inducing, len(X))
        kmeans = MiniBatchKMeans(n_clusters=n_inducing, n_init=3, random_state=0, batch_size=1000)
        kmeans.fit(X.numpy())
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        self.model = FastSVGP(inducing_points.to(self.device)).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.lr)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(y))
        loader = DataLoader(TensorDataset(X, y), batch_size=512, shuffle=True)

        for epoch in range(self.num_epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = -mll(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.likelihood.eval()

    def predict(self, X, batch_size=4096):
        X = torch.as_tensor(X, dtype=torch.float32)
        means, stds = [], []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(self.device)
                pred = self.likelihood(self.model(X_batch))
                means.append(pred.mean.cpu())
                stds.append(pred.variance.sqrt().cpu())

        return torch.cat(means).numpy(), torch.cat(stds).numpy()


def acq_uncertainty(surrogate, X):
    _, std = surrogate.predict(X)
    return std

def acq_ucb(surrogate, X, beta=2.0):
    mean, std = surrogate.predict(X)
    return mean + beta * std

def acq_lcb(surrogate, X, beta=2.0):
    mean, std = surrogate.predict(X)
    return -mean + beta * std

def acq_random(surrogate, X):
    return np.random.rand(len(X))


ACQUISITION_FNS = {
    'uncertainty': acq_uncertainty,
    'ucb': lambda s, X: acq_ucb(s, X, beta=2.0),
    'lcb': lambda s, X: acq_lcb(s, X, beta=2.0),
    'random': acq_random,
}


def compute_metrics(J):
    J_sum = J.sum(axis=1)
    
    return {
        'mean_|J|': np.abs(J).mean(),
        'mean_|J_sum|': np.abs(J_sum).mean(),
        'std_J': J.std(),
        'frac_|J|>0.7': (np.abs(J) > 0.7).mean(),
        'frac_|J|>0.8': (np.abs(J) > 0.8).mean(),
        'frac_|J|>0.9': (np.abs(J) > 0.9).mean(),
        'n_|Jsum|>4': (np.abs(J_sum) > 4.0).sum(),
        'n_|Jsum|>5': (np.abs(J_sum) > 5.0).sum(),
        'n_all|J|>0.8': ((np.abs(J) > 0.8).all(axis=1)).sum(),
    }


def run_al_test(X_pool, y_pool, acq_name, acq_fn, n_initial=500, n_rounds=15, n_select=200,
                num_inducing=150, num_epochs=20):
    
    print(f"\n{'='*70}")
    print(f"ACQUISITION: {acq_name.upper()}")
    print(f"{'='*70}")
    
    rng = np.random.default_rng(42)
    
    J_pool = X_pool[:, 4:10]
    
    available = np.ones(len(X_pool), dtype=bool)
    init_idx = rng.choice(len(X_pool), size=n_initial, replace=False)
    available[init_idx] = False
    train_idx = list(init_idx)
    
    round_metrics = []
    cumulative_metrics = []
    timings = []
    
    init_metrics = compute_metrics(J_pool[train_idx])
    print(f"Initial: {n_initial} samples")
    print(f"  mean|J|={init_metrics['mean_|J|']:.4f}  frac|J|>0.8={init_metrics['frac_|J|>0.8']:.4f}")
    print()
    print(f"{'Round':<6} {'mean|J|':<10} {'frac>0.8':<10} {'n|Js|>4':<10} {'n|Js|>5':<10} {'time':<8}")
    print("-" * 60)
    
    for r in range(n_rounds):
        t0 = time.time()
        
        surrogate = FastSurrogate(num_inducing=num_inducing, num_epochs=num_epochs)
        surrogate.fit(J_pool[train_idx], y_pool[train_idx, 0])
        t_fit = time.time() - t0
        
        cand_idx = np.where(available)[0]
        
        t1 = time.time()
        scores = acq_fn(surrogate, J_pool[cand_idx])
        t_score = time.time() - t1
        
        top_local = np.argsort(scores)[-n_select:]
        top_idx = cand_idx[top_local]
        
        available[top_idx] = False
        train_idx.extend(top_idx)
        
        selected_J = J_pool[top_idx]
        metrics = compute_metrics(selected_J)
        round_metrics.append(metrics)
        
        cumul = compute_metrics(J_pool[train_idx])
        cumulative_metrics.append(cumul)
        
        timings.append({'fit': t_fit, 'score': t_score})
        
        print(f"{r+1:<6} {metrics['mean_|J|']:<10.4f} {metrics['frac_|J|>0.8']:<10.4f} "
              f"{metrics['n_|Jsum|>4']:<10} {metrics['n_|Jsum|>5']:<10} {t_fit+t_score:<8.1f}s")
    
    print("-" * 60)
    final_metrics = compute_metrics(J_pool[train_idx])
    print(f"FINAL: {len(train_idx)} samples")
    print(f"  mean|J|={final_metrics['mean_|J|']:.4f}  frac|J|>0.8={final_metrics['frac_|J|>0.8']:.4f}")
    print(f"  n|Jsum|>4={final_metrics['n_|Jsum|>4']}  n|Jsum|>5={final_metrics['n_|Jsum|>5']}  n_all|J|>0.8={final_metrics['n_all|J|>0.8']}")
    
    return {
        'train_idx': train_idx,
        'round_metrics': round_metrics,
        'cumulative_metrics': cumulative_metrics,
        'timings': timings,
        'final_metrics': final_metrics,
    }


def print_comparison_table(results):
    print("\n")
    print("=" * 90)
    print("COMPARISON TABLE: Per-Round Selected Points")
    print("=" * 90)
    
    acq_names = list(results.keys())
    n_rounds = len(results[acq_names[0]]['round_metrics'])
    
    print(f"\n--- mean|J| per round (higher = more extreme) ---")
    header = f"{'Round':<8}" + "".join([f"{name:<15}" for name in acq_names])
    print(header)
    print("-" * len(header))
    for r in range(n_rounds):
        row = f"{r+1:<8}"
        for name in acq_names:
            val = results[name]['round_metrics'][r]['mean_|J|']
            row += f"{val:<15.4f}"
        print(row)
    
    print(f"\n--- frac|J|>0.8 per round (higher = more boundary) ---")
    header = f"{'Round':<8}" + "".join([f"{name:<15}" for name in acq_names])
    print(header)
    print("-" * len(header))
    for r in range(n_rounds):
        row = f"{r+1:<8}"
        for name in acq_names:
            val = results[name]['round_metrics'][r]['frac_|J|>0.8']
            row += f"{val:<15.4f}"
        print(row)
    
    print(f"\n--- n|Jsum|>4 per round (count of boundary-ish points) ---")
    header = f"{'Round':<8}" + "".join([f"{name:<15}" for name in acq_names])
    print(header)
    print("-" * len(header))
    for r in range(n_rounds):
        row = f"{r+1:<8}"
        for name in acq_names:
            val = results[name]['round_metrics'][r]['n_|Jsum|>4']
            row += f"{val:<15}"
        print(row)


def print_final_summary(results):
    print("\n")
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    
    metrics = ['mean_|J|', 'frac_|J|>0.8', 'frac_|J|>0.9', 'n_|Jsum|>4', 'n_|Jsum|>5', 'n_all|J|>0.8']
    acq_names = list(results.keys())
    
    header = f"{'Metric':<20}" + "".join([f"{name:<15}" for name in acq_names])
    print(header)
    print("-" * len(header))
    
    for metric in metrics:
        row = f"{metric:<20}"
        for name in acq_names:
            val = results[name]['final_metrics'][metric]
            if isinstance(val, float):
                row += f"{val:<15.4f}"
            else:
                row += f"{val:<15}"
        print(row)
    
    print("\n--- Timing (avg seconds per round) ---")
    header = f"{'Component':<20}" + "".join([f"{name:<15}" for name in acq_names])
    print(header)
    print("-" * len(header))
    
    row_fit = f"{'fit':<20}"
    row_score = f"{'score':<20}"
    row_total = f"{'total':<20}"
    
    for name in acq_names:
        times = results[name]['timings']
        avg_fit = np.mean([t['fit'] for t in times])
        avg_score = np.mean([t['score'] for t in times])
        row_fit += f"{avg_fit:<15.1f}"
        row_score += f"{avg_score:<15.1f}"
        row_total += f"{avg_fit+avg_score:<15.1f}"
    
    print(row_fit)
    print(row_score)
    print(row_total)


def print_histogram(J, name, bins=10):
    J_sum = J.sum(axis=1)
    
    print(f"\n--- J_sum Histogram for {name} ---")
    
    hist, edges = np.histogram(J_sum, bins=bins, range=(-6, 6))
    max_count = max(hist)
    scale = 40 / max_count if max_count > 0 else 1
    
    for i in range(len(hist)):
        lo, hi = edges[i], edges[i+1]
        bar = "█" * int(hist[i] * scale)
        print(f"  [{lo:>5.1f},{hi:>5.1f}): {hist[i]:>5} {bar}")
    
    print(f"  |J_sum| > 4: {(np.abs(J_sum) > 4).sum()}")
    print(f"  |J_sum| > 5: {(np.abs(J_sum) > 5).sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to train.pt')
    parser.add_argument('--n_pool', type=int, default=50000, help='Pool size')
    parser.add_argument('--n_initial', type=int, default=500, help='Initial training size')
    parser.add_argument('--n_rounds', type=int, default=15, help='AL rounds')
    parser.add_argument('--n_select', type=int, default=200, help='Select per round')
    parser.add_argument('--num_inducing', type=int, default=150, help='SVGP inducing points')
    parser.add_argument('--num_epochs', type=int, default=20, help='SVGP epochs')
    
    args = parser.parse_args()
    
    print("=" * 90)
    print("ACTIVE LEARNING TEST BENCH")
    print("=" * 90)
    print(f"Data: {args.data}")
    print(f"Pool: {args.n_pool}, Initial: {args.n_initial}, Rounds: {args.n_rounds}, Select: {args.n_select}")
    print(f"SVGP: {args.num_inducing} inducing, {args.num_epochs} epochs")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    print("\nLoading data...")
    data = torch.load(args.data, weights_only=False)
    X_all = data['X'].numpy()
    y_all = data['y'].numpy()
    
    print(f"Total data: {len(X_all)}")
    
    rng = np.random.default_rng(42)
    pool_idx = rng.choice(len(X_all), size=min(args.n_pool, len(X_all)), replace=False)
    X_pool = X_all[pool_idx]
    y_pool = y_all[pool_idx]
    
    print(f"Pool size: {len(X_pool)}")
    
    J_pool = X_pool[:, 4:10]
    pool_metrics = compute_metrics(J_pool)
    print(f"\nPool stats:")
    print(f"  mean|J|={pool_metrics['mean_|J|']:.4f}")
    print(f"  frac|J|>0.8={pool_metrics['frac_|J|>0.8']:.4f}")
    print(f"  n|Jsum|>4={pool_metrics['n_|Jsum|>4']}")
    print(f"  n|Jsum|>5={pool_metrics['n_|Jsum|>5']}")
    
    print_histogram(J_pool, "POOL")
    
    results = {}
    for name, acq_fn in ACQUISITION_FNS.items():
        results[name] = run_al_test(
            X_pool, y_pool, name, acq_fn,
            n_initial=args.n_initial,
            n_rounds=args.n_rounds,
            n_select=args.n_select,
            num_inducing=args.num_inducing,
            num_epochs=args.num_epochs,
        )
    
    print_comparison_table(results)
    print_final_summary(results)
    
    print("\n")
    print("=" * 90)
    print("J_SUM HISTOGRAMS (Final Training Sets)")
    print("=" * 90)
    for name, res in results.items():
        J_selected = J_pool[res['train_idx']]
        print_histogram(J_selected, name.upper())


if __name__ == '__main__':
    main()
