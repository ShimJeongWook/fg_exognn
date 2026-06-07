import os
import random
import sys

import numpy as np

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

if hasattr(torch.utils, "_pytree") and (
    not hasattr(torch.utils._pytree, "register_pytree_node")
    and hasattr(torch.utils._pytree, "_register_pytree_node")
):
    def _compat_register_pytree_node(
        typ,
        flatten_fn,
        unflatten_fn,
        *,
        to_dumpable_context=None,
        from_dumpable_context=None,
        **kwargs,
    ):
        return torch.utils._pytree._register_pytree_node(
            typ,
            flatten_fn,
            unflatten_fn,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context,
        )

    torch.utils._pytree.register_pytree_node = _compat_register_pytree_node


from src.engines.deepair_engine import DeepAirEngine
from src.models.fg_exognn import ExoGNN
from src.utils.args import get_public_config
from src.utils.dataloader_deepair import get_dataset_info, load_adj_from_numpy, load_dataset
from src.utils.logging import get_logger
from src.utils.metrics import masked_mae, masked_mape, masked_rmse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patch_len', type=int, default=12)
    parser.add_argument('--patch_stride', type=int, default=12)
    parser.add_argument('--target_var_indices', type=str, default=None)
    parser.add_argument('--exogenous_var_indices', type=str, default=None)
    parser.add_argument('--cheb_k', type=int, default=3)
    parser.add_argument('--restrict_static_graph', type=int, default=0)
    parser.add_argument('--use_spatial', type=int, default=1)
    parser.add_argument('--use_exo_attn', type=int, default=1)
    parser.add_argument('--use_cross', type=int, default=1)
    parser.add_argument('--use_future_exo', type=int, default=1)
    parser.add_argument('--use_exo_head_context', type=int, default=1)
    parser.add_argument('--use_future_step_context', type=int, default=1)
    parser.add_argument('--use_amp', type=int, default=1)

    parser.add_argument('--lrate', type=float, default=3e-4)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    parser.add_argument(
        '--loss',
        type=str,
        default='mae',
        choices=['mae', 'mae_rmse_mape'],
    )
    parser.add_argument('--rmse_loss_weight', type=float, default=0.05)
    parser.add_argument('--mape_loss_weight', type=float, default=5.0)
    parser.add_argument('--exp_tag', type=str, default='')

    args = parser.parse_args()
    if args.exp_tag:
        log_dir = './experiments/{}/{}/sweep/{}/'.format(args.model_name, args.dataset, args.exp_tag)
    else:
        log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    return args, log_dir, logger


def build_loss(args):
    def loss_fn(preds, labels, null_val=np.nan):
        if args.loss == 'mae':
            return masked_mae(preds, labels, null_val)
        mae = masked_mae(preds, labels, null_val)
        rmse = masked_rmse(preds, labels, null_val)
        mape = masked_mape(preds, labels, null_val)
        return mae + args.rmse_loss_weight * rmse + args.mape_loss_weight * mape

    return loss_fn


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)
    adj_mx = load_adj_from_numpy(adj_path)
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = ExoGNN(
        adj_mx=adj_mx,
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        target_var_indices=args.target_var_indices,
        exogenous_var_indices=args.exogenous_var_indices,
        use_spatial=args.use_spatial,
        use_exo_attn=args.use_exo_attn,
        use_cross=args.use_cross,
        cheb_k=args.cheb_k,
        use_future_exo=args.use_future_exo,
        use_exo_head_context=args.use_exo_head_context,
        use_future_step_context=args.use_future_step_context,
        restrict_static_graph=args.restrict_static_graph,
    )

    loss_fn = build_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    args.lr_update_in_step = 0
    engine = DeepAirEngine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        args=args,
    )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
