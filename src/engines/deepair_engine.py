import os
import time
import torch
import numpy as np
from tqdm import tqdm
from src.utils.metrics import masked_mape, masked_f1_score
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics
import sys
from src.base.engine import BaseEngine

class DeepAirEngine(BaseEngine):
    def __init__(self, **args):
        super(DeepAirEngine, self).__init__(**args)

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in tqdm(self._dataloader['train_loader'].get_iterator()):
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X[..., :self._args.input_dim], label]))
            pred_n = self.model(X, label[..., 1:])
            label_n = label[..., 0:1]
            pred, label = self._inverse_transform([pred_n, label_n])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            if os.environ.get('DEV_NORM_LOSS') == '1':
                # compute the loss on the NORMALIZED scale (like PM2.5-GNN) so that
                # MSE-style losses are well-scaled and do not collapse to the mean.
                std0 = self._scaler.std.flatten()[0].to(pred_n.device)
                mean0 = self._scaler.mean.flatten()[0].to(pred_n.device)
                nv = (mask_value.to(pred_n.device).float() - mean0) / std0
                loss = self._loss_fn(pred_n, label_n, nv)
            else:
                loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), None


    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X[..., :self._args.input_dim], label]))
                pred = self.model(X, label[..., 1:])
                pred, label = self._inverse_transform([pred, label[..., 0:1]])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))

            f1_score = masked_f1_score(preds, labels)
            log = 'F1 Score for level 0: {:.4f}, 1: {:.4f}, 2: {:.4f}'
            self._logger.info(log.format(f1_score[0], f1_score[1], f1_score[2]))

            # ---- ALSO compute metrics with the ORIGINAL PM2.5-GNN get_metric formula ----
            try:
                p = preds.numpy(); l = labels.numpy()
                hz = 75
                hit = np.sum((p>=hz)&(l>=hz)); miss = np.sum((l>=hz)&(p<hz)); fa = np.sum((p>=hz)&(l<hz))
                csi = hit/(hit+fa+miss+1e-12); pod = hit/(hit+miss+1e-12); far = fa/(hit+fa+1e-12)
                pp = np.transpose(p,(0,2,1)).reshape((-1,p.shape[1]))
                ll = np.transpose(l,(0,2,1)).reshape((-1,l.shape[1]))
                rmse_o = np.mean(np.sqrt(np.mean(np.square(pp-ll),axis=1)))
                mae_o = np.mean(np.mean(np.abs(pp-ll),axis=1))
                self._logger.info('[PM2.5-GNN-style] RMSE: {:.4f}, MAE: {:.4f}, CSI: {:.4f}, POD: {:.4f}, FAR: {:.4f}'.format(
                    rmse_o, mae_o, csi, pod, far))
                # ---- R2 and IOA (Willmott index of agreement), overall + per-horizon ----
                def _r2_ioa(pr, ob):
                    pr = pr.ravel().astype('float64'); ob = ob.ravel().astype('float64')
                    obar = ob.mean()
                    ss_res = np.sum((ob-pr)**2); ss_tot = np.sum((ob-obar)**2)
                    r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else float('nan')
                    den = np.sum((np.abs(pr-obar)+np.abs(ob-obar))**2)
                    ioa = 1.0 - ss_res/den if den > 1e-12 else float('nan')
                    return r2, ioa
                r2_all, ioa_all = _r2_ioa(p, l)
                self._logger.info('[R2-IOA] R2: {:.4f}, IOA: {:.4f}'.format(r2_all, ioa_all))
                for i in range(p.shape[1]):
                    r2h, ioah = _r2_ioa(p[:, i, :], l[:, i, :])
                    self._logger.info('[R2-IOA] Horizon {:d}, R2: {:.4f}, IOA: {:.4f}'.format(i+1, r2h, ioah))
            except Exception as e:
                self._logger.info('PM2.5-GNN-style metric skipped: {}'.format(e))

    # def corr(self, mode):
    #     # self.model.eval()

    #     Xs = []
    #     Ys = []
    #     Zp = []
    #     Zf = []
    #     # preds = []
    #     # labels = []
    #     with torch.no_grad():
    #         for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
    #             # X (b, t, n, f), label (b, t, n, 1)
    #             X, label = self._to_device(self._to_tensor([X[..., :self._args.input_dim], label]))
                
    #             Xs.append(X[..., 0:1].cpu())
    #             Ys.append(X[..., 1:].cpu())
    #             Zp.append(label.cpu())
    #             Zf.append(label.cpu())
    #             # pred = self.model(X, label[..., 1:])
    #             # pred, label = self._inverse_transform([pred, label[..., 0:1]])

    #             # preds.append(pred.squeeze(-1).cpu())
    #             # labels.append(label.squeeze(-1).cpu())

    #     preds = torch.cat(preds, dim=0)
    #     labels = torch.cat(labels, dim=0)
    #     mask_value = torch.tensor(0)
    #     if labels.min() < 1:
    #         mask_value = labels.min()

    #     if mode == 'val':
    #         mae = self._loss_fn(preds, labels, mask_value).item()
    #         mape = masked_mape(preds, labels, mask_value).item()
    #         rmse = masked_rmse(preds, labels, mask_value).item()
    #         return mae, mape, rmse

    #     elif mode == 'test':
    #         test_mae = []
    #         test_mape = []
    #         test_rmse = []
    #         print('Check mask value', mask_value)
    #         for i in range(self.model.horizon):
    #             res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
    #             log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    #             self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
    #             test_mae.append(res[0])
    #             test_mape.append(res[1])
    #             test_rmse.append(res[2])

    #         log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    #         self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))

    #         f1_score = masked_f1_score(preds, labels)
    #         log = 'F1 Score for level 0: {:.4f}, 1: {:.4f}, 2: {:.4f}'
    #         self._logger.info(log.format(f1_score[0], f1_score[1], f1_score[2]))