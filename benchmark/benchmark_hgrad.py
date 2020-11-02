import argparse
import datetime

import numpy as np
import torch
import pandas as pd

from falkon import FalkonOptions
from falkon.center_selection import FixedSelector
from falkon.hypergrad.falkon_ho import run_falkon_hypergrad, map_gradient

from datasets import get_load_fn, equal_split
from benchmark_utils import *
from error_metrics import get_err_fns


def run_gmap_exp(dataset: Dataset,
                 sigma_type: str,
                 inner_maxiter: int):
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float32, as_torch=True)
    err_fns = get_err_fns(dataset)
    centers = torch.from_numpy(metadata['centers']).cuda()

    # We use a validation split (redefinition of Xtr, Ytr).
    train_frac = 0.8
    idx_tr, idx_val = equal_split(Xtr.shape[0], train_frac=train_frac)
    Xval, Yval = Xtr[idx_val], Ytr[idx_val]
    Xtr, Ytr = Xtr[idx_tr], Ytr[idx_tr]
    print("Splitting data for validation and testing: Have %d train - %d validation samples" %
          (Xtr.shape[0], Xval.shape[0]))
    data = {'Xtr': Xtr.cuda(), 'Ytr': Ytr.cuda(), 'Xts': Xval.cuda(), 'Yts': Yval.cuda()}

    falkon_opt = FalkonOptions(use_cpu=False)

    df: pd.DataFrame = map_gradient(data,
                      falkon_centers=FixedSelector(centers),
                      falkon_M=centers.shape[0],
                      falkon_maxiter=inner_maxiter,
                      falkon_opt=falkon_opt,
                      sigma_type=sigma_type
                      )
    out_fn = f"./gd_map_{dataset}_{int(datetime.datetime.timestamp(datetime.datetime.now()) * 1000)}.csv"
    print("Saving gradient map to %s" % (out_fn))
    df.to_csv(out_fn)


def run_exp(dataset: Dataset,
            inner_maxiter: int,
            outer_lr: float,
            outer_steps: int,
            hessian_cg_steps: int,
            hessian_cg_tol: float,
            sigma_type: str,
            opt_centers: bool):
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float32, as_torch=True)
    err_fns = get_err_fns(dataset)

    centers = torch.from_numpy(metadata['centers']).cuda()

    # We use a validation split (redefinition of Xtr, Ytr).
    train_frac = 0.8
    idx_tr, idx_val = equal_split(Xtr.shape[0], train_frac=train_frac)
    Xval, Yval = Xtr[idx_val], Ytr[idx_val]
    Xtr, Ytr = Xtr[idx_tr], Ytr[idx_tr]
    print("Splitting data for validation and testing: Have %d train - %d validation samples" %
          (Xtr.shape[0], Xval.shape[0]))
    data = {'Xtr': Xtr.cuda(), 'Ytr': Ytr.cuda(), 'Xts': Xval.cuda(), 'Yts': Yval.cuda()}

    falkon_opt = FalkonOptions(use_cpu=False)

    def cback(model):
        train_pred = model.predict(data['Xtr'])
        val_pred = model.predict(data['Xts'])
        train_err, err = err_fns[0](data['Ytr'].cpu(), train_pred.cpu(), **metadata)
        val_err, err = err_fns[0](data['Yts'].cpu(), val_pred.cpu(), **metadata)
        print(f"Train {err}: {train_err:.5f} -- Val {err}: {val_err:.5f}")

    hps, val_loss, hgrads, best_model, times = run_falkon_hypergrad(
        data,
        sigma_type=sigma_type,
        falkon_M=centers.shape[0],
        falkon_centers=FixedSelector(centers),
        optimize_centers=opt_centers,
        falkon_maxiter=inner_maxiter,
        falkon_opt=falkon_opt,
        outer_lr=outer_lr,
        outer_steps=outer_steps,
        hessian_cg_steps=hessian_cg_steps,
        hessian_cg_tol=hessian_cg_tol,
        callback=cback,
        debug=True,
    )

    # Now we have the model, retrain with the full training data and test!
    Xtr = torch.cat([Xtr, Xval], 0).cuda()
    Ytr = torch.cat([Ytr, Yval], 0).cuda()
    best_model.fit(Xtr, Ytr)
    train_pred = best_model.predict(Xtr).cpu()
    test_pred = best_model.predict(Xts.cuda()).cpu()

    print("Test (unseen) errors after retraining on the full train dataset")
    for efn in err_fns:
        train_err, err = efn(Ytr.cpu(), train_pred, **metadata)
        test_err, err = efn(Yts.cpu(), test_pred, **metadata)
        print(f"Train {err}: {train_err:.5f} -- Test {err}: {test_err:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")

    p.add_argument('-d', '--dataset', type=Dataset, choices=list(Dataset), required=True,
                   help='Dataset')
    p.add_argument('-s', '--seed', type=int, required=True, help="Random seed")
    p.add_argument('--flk-steps', type=int, help="Maximum number of Falkon steps",
                   default=10)
    p.add_argument('--lr', type=float, help="Learning rate for the outer-problem solver",
                   default=0.01)
    p.add_argument('--steps', type=int, help="Number of outer-problem steps",
                   default=100)
    p.add_argument('--hessian-cg-steps', type=int,
                   help="Maximum steps for finding the Hessian via CG",
                   default=10)
    p.add_argument('--hessian-cg-tol', type=float, help="Tolerance for Hessian CG problem",
                   default=1e-4)
    p.add_argument('--sigma-type', type=str,
                   help="Use diagonal or single lengthscale for the kernel",
                   default='single')
    p.add_argument('--optimize-centers', action='store_true',
                   help="Whether to optimize Nystrom centers")
    p.add_argument('--map-gradient', action='store_true', help="Creates a gradient map")

    args = p.parse_args()
    print("-------------------------------------------")
    print(print(datetime.datetime.now()))
    print("############### SEED: %d ################" % (args.seed))
    print("-------------------------------------------")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.map_gradient:
        run_gmap_exp()

    run_exp(dataset=args.dataset,
            inner_maxiter=args.flk_steps,
            outer_lr=args.lr,
            outer_steps=args.steps,
            hessian_cg_steps=args.hessian_cg_steps,
            hessian_cg_tol=args.hessian_cg_tol,
            sigma_type=args.sigma_type,
            opt_centers=args.optimize_centers)
