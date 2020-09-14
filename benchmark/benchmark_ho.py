import argparse
import time
import datetime

import torch
import numpy as np

from benchmark_utils import *
from datasets import get_load_fn, get_cv_fn, equal_split
from error_metrics import get_err_fns, get_tf_err_fn

from falkon import kernels
from falkon.models import falkon
from falkon.utils import TicToc
from falkon.hypergrad.falkon_ho import run_falkon_hypergrad


RANDOM_SEED = 1219


def run_on_dataset(dset: Dataset,
                   dtype: Optional[DataType],
                   seed: int,
                   inner_iter: int,
                   outer_iter: int,
                   outer_lr: float,
                   num_centers: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data types
    if dtype is None:
        dtype = DataType.float64

    # Data load
    load_fn = get_load_fn(dset)
    err_fns = get_err_fns(dset)
    Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True)
    Xtr = Xtr.pin_memory()
    Ytr = Ytr.pin_memory()
    # Further split the data into train/validation
    idx_tr, idx_val = equal_split(Xtr.shape[0], 0.8)
    Xval, Yval = Xtr[idx_val,:], Ytr[idx_val,:]
    Xtr, Ytr = Xtr[idx_tr,:], Ytr[idx_tr,:]
    data = {'Xtr': Xtr, 'Ytr': Ytr, 'Xts': Xval, 'Yts': Yval}

    # Options and parameters
    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=True,
        pc_epsilon_32=1e-6,
        pc_epsilon_64=1e-13,
        debug=False
    )
    hessian_cg_steps = 20
    hessian_cg_tol = 1e-3
    debug = True

    with TicToc("Hyperparameter optimization"):
        out_dict = run_falkon_hypergrad(data=data,
                                        falkon_M=num_centers,
                                        falkon_maxiter=inner_iter,
                                        falkon_opt=opt,
                                        outer_lr=outer_lr,
                                        outer_steps=outer_iter,
                                        hessian_cg_steps=hessian_cg_steps,
                                        hessian_cg_tol=hessian_cg_tol,
                                        debug=True)
    hparam_history, val_loss_history, hgrad_history, model = out_dict
    # Refit model
    model.fit(Xtr, Ytr)
    test_preds = model.predict(Xts)
    for err_fn in err_fns:
        test_error, err_name = err_fn(Yts, test_preds, **kwargs)
    print(f"Test {err_name} = {test_error:.5f}")


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Falkon HyperOpt Runner")

    p.add_argument('-d', '--dataset', type=Dataset, choices=list(Dataset), required=True,
                   help='Dataset')
    p.add_argument('-t', '--dtype', type=DataType.argparse, choices=list(DataType),
                   required=False, default=None,
                   help='Floating point precision to work with. Lower precision will be '
                        'faster but less accurate. Certain algorithms require a specific precision. '
                        'If this argument is not specified we will use the highest precision '
                        'supported by the chosen algorithm.')
    p.add_argument('-i', '--inner-iter', type=int, required=True,
                   help='Number of inner CG iterations to run Falkon')
    p.add_argument('-o', '--outer-iter', type=int, required=True,
                   help='Number of hyperparameter gradient descent iterations')
    p.add_argument('--seed', type=int, default=RANDOM_SEED,
                    help='Random number generator seed')
    # Algorithm-specific arguments
    p.add_argument('-M', '--num-centers', type=int, required=True,
                        help='Number of Nystroem centers. Used for algorithms '
                              'falkon, gpytorch and gpflow.')
    p.add_argument('--outer-lr', type=float, default=0.1, help="Outer gradient descent learning rate")

    args = p.parse_args()
    print("-------------------------------------------")
    print(f"STARTING AT {datetime.datetime.now()} -- SEED={args.seed}")

    run_on_dataset(dset=args.dataset, dtype=args.dtype, seed=args.seed,
                   inner_iter=args.inner_iter, outer_iter=args.outer_iter,
                   outer_lr=args.outer_lr, num_centers=args.num_centers)
