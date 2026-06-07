import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.base.model import BaseModel


def _parse_indices(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return list(default)
        return [int(x.strip()) for x in value.split(",") if x.strip() != ""]
    return [int(x) for x in value]


def _scale_to_logit(scale):
    scale = min(max(float(scale), 1e-6), 1.0 - 1e-6)
    return math.log(scale / (1.0 - scale))


class SpatialPositionalEncoding(nn.Module):
    """2D Spatial Positional Encoding based on latitude and longitude."""

    def __init__(self, d_model, coords=None):
        super().__init__()
        self.d_model = d_model

        if coords is not None:
            pe = self._compute_positional_encoding(coords, d_model)
            self.register_buffer("pe", pe)
        else:
            self.pe = None

    @staticmethod
    def _compute_positional_encoding(coords, d_model):
        n_nodes = coords.shape[0]
        pe = np.zeros((n_nodes, d_model), dtype=np.float32)

        lats = coords[:, 0]
        lons = coords[:, 1]

        n_groups = d_model // 4

        for i in range(n_groups):
            freq = 10000.0 ** (2 * i / d_model)
            pe[:, 4 * i] = np.sin(lats / freq)
            pe[:, 4 * i + 1] = np.sin(lons / freq)
            pe[:, 4 * i + 2] = np.cos(lats / freq)
            pe[:, 4 * i + 3] = np.cos(lons / freq)

        remaining = d_model % 4
        if remaining > 0:
            base_idx = n_groups * 4
            freq = 10000.0 ** (2 * n_groups / d_model)
            if remaining >= 1:
                pe[:, base_idx] = np.sin(lats / freq)
            if remaining >= 2:
                pe[:, base_idx + 1] = np.sin(lons / freq)
            if remaining >= 3:
                pe[:, base_idx + 2] = np.cos(lats / freq)

        return torch.from_numpy(pe)

    def forward(self, n_nodes=None):
        if self.pe is None:
            raise ValueError("Positional encoding not initialized.")
        if n_nodes is not None and n_nodes != self.pe.shape[0]:
            raise ValueError(f"Expected {self.pe.shape[0]} nodes, got {n_nodes}")
        return self.pe


class VariatePositionalEncoding(nn.Module):
    """Learnable variate (channel) positional encoding."""

    def __init__(self, n_variates, d_model):
        super().__init__()
        self.ce = nn.Parameter(torch.zeros(n_variates, d_model))
        nn.init.normal_(self.ce, std=0.02)

    def forward(self, variate_indices=None):
        if variate_indices is not None:
            return self.ce[variate_indices]
        return self.ce


class TargetPatchEmbedding(nn.Module):
    """Target endogenous branch with temporal patching."""

    def __init__(self, seq_len, patch_len, patch_stride, d_model, dropout=0.1):
        super().__init__()
        if patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if patch_len > seq_len:
            raise ValueError(f"patch_len={patch_len} cannot exceed seq_len={seq_len}")

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_patches = (seq_len - patch_len) // patch_stride + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, O, T = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {T}")

        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        patches = patches.contiguous()

        z = self.patch_proj(patches)
        return self.dropout(z)


class TargetPatchTemporalAttention(nn.Module):
    """Self-attention over temporal patch tokens P for each node-target pair."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        B, N, O, P, D = h.shape
        if P <= 1:
            return h
        residual = h
        z = self.norm(h).contiguous().view(B * N * O, P, D)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = z.view(B, N, O, P, D)
        return residual + self.dropout(z)


class ExogenousEmbedding(nn.Module):
    """Historical + future exogenous variable branch."""

    def __init__(self, seq_len, pred_len, d_model, dropout=0.1, use_future=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_future = use_future
        in_len = seq_len + (pred_len if use_future else 0)
        self.time_proj = nn.Linear(in_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, future=None):
        B, N, E, _ = x.shape
        if E == 0:
            return x.new_zeros(B, N, 0, self.time_proj.out_features)

        if self.use_future:
            if future is None:
                future = x.new_zeros(B, N, E, self.pred_len)
            x = torch.cat([x, future], dim=-1)

        z = self.time_proj(x)
        return self.dropout(z)


class ExogenousVariableAttention(nn.Module):
    """Self-attention over exogenous variable tokens E."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        B, N, E, D = h.shape
        if E <= 1:
            return h
        residual = h
        z = self.norm(h).contiguous().view(B * N, E, D)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = z.view(B, N, E, D)
        return residual + self.dropout(z)


class ExoTargetPatchCrossAttention(nn.Module):
    """Target patch tokens attend to exogenous variable tokens station-locally."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, exo):
        B, N, O, P, D = target.shape
        E = exo.shape[2]
        if E == 0:
            return target

        q = self.q_norm(target).contiguous().view(B * N, O * P, D)
        kv = self.kv_norm(exo).contiguous().view(B * N, E, D)
        z, _ = self.attn(q, kv, kv, need_weights=False)
        z = z.view(B, N, O, P, D)
        return target + self.dropout(z)


class AdaptiveChebyshevGraphConv(nn.Module):
    """Chebyshev graph convolution over station nodes using static and adaptive supports."""

    def __init__(self, n_nodes, d_model, cheb_k=3, dropout=0.1,
                 restrict_static_graph=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.cheb_k = max(1, int(cheb_k))
        self.restrict_static_graph = bool(restrict_static_graph)

        support_len = 2
        in_dim = (1 + support_len * max(self.cheb_k - 1, 0)) * d_model

        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(in_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.static_gate = nn.Parameter(
            torch.tensor(_scale_to_logit(0.9), dtype=torch.float32)
        )
        self.adaptive_gate = nn.Parameter(
            torch.tensor(_scale_to_logit(0.1), dtype=torch.float32)
        )
        self.residual_gate = nn.Parameter(
            torch.tensor(_scale_to_logit(0.2), dtype=torch.float32)
        )

    @staticmethod
    def _normalize_support(support):
        support = support / support.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.nan_to_num(support, nan=0.0)

    @staticmethod
    def _propagate(support, x):
        return torch.einsum("nm,bmd->bnd", support, x)

    def _build_supports(self, A_static, A_adaptive):
        A_static = A_static.to(device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)
        static_mask = A_static > 1e-8
        A_static = self._normalize_support(A_static)

        if self.restrict_static_graph:
            A_adaptive = A_adaptive.masked_fill(~static_mask, 0.0)
        A_adaptive = A_adaptive.to(device=A_static.device, dtype=A_static.dtype)
        A_adaptive = self._normalize_support(A_adaptive)

        return [
            torch.sigmoid(self.static_gate) * A_static,
            torch.sigmoid(self.adaptive_gate) * A_adaptive,
        ]

    def forward(self, h, A_static, A_adaptive):
        B, N, O, P, D = h.shape
        residual = h

        x = self.norm(h)
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(B * O * P, N, D)

        terms = [x]
        if A_adaptive is None:
            A_adaptive = torch.eye(N, device=x.device, dtype=x.dtype)
        supports = self._build_supports(A_static, A_adaptive)

        if self.cheb_k > 1:
            for support in supports:
                x_prev_prev = x
                x_prev = self._propagate(support, x)
                terms.append(x_prev)
                for _ in range(2, self.cheb_k):
                    x_next = 2.0 * self._propagate(support, x_prev) - x_prev_prev
                    terms.append(x_next)
                    x_prev_prev, x_prev = x_prev, x_next

        out = self.out_proj(torch.cat(terms, dim=-1))
        out = out.view(B, O, P, N, D).permute(0, 3, 1, 2, 4)
        return residual + torch.sigmoid(self.residual_gate) * self.dropout(out)

    def get_support_gates(self):
        with torch.no_grad():
            return {
                "static_gate": torch.sigmoid(self.static_gate).item(),
                "adaptive_gate": torch.sigmoid(self.adaptive_gate).item(),
                "residual_gate": torch.sigmoid(self.residual_gate).item(),
                "cheb_k": self.cheb_k,
            }


class PositionwiseFFN(nn.Module):
    """FFN over the last dimension D."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        residual = h
        z = self.fc2(self.dropout(self.act(self.fc1(self.norm(h)))))
        return residual + self.dropout(z)


class PatchXerGNNBlock(nn.Module):
    """
    Patch-TimeXer-GNN block:
    target patch temporal attention -> FFN
    -> Chebyshev spatial convolution -> FFN
    -> exogenous variable attention -> FFN
    -> exo-to-target patch cross-attention -> FFN
    """

    def __init__(self, n_nodes, d_model, n_heads, d_ff, dropout,
                 use_exo_attn=True, use_cross=True, use_spatial=True,
                 cheb_k=3, restrict_static_graph=False):
        super().__init__()
        self.use_exo_attn = use_exo_attn
        self.use_cross = use_cross
        self.use_spatial = use_spatial

        self.target_temporal = TargetPatchTemporalAttention(d_model, n_heads, dropout)
        self.target_temporal_ffn = PositionwiseFFN(d_model, d_ff, dropout)

        if use_exo_attn:
            self.exo_attn = ExogenousVariableAttention(d_model, n_heads, dropout)
            self.exo_attn_ffn = PositionwiseFFN(d_model, d_ff, dropout)

        if use_cross:
            self.cross_attn = ExoTargetPatchCrossAttention(d_model, n_heads, dropout)
            self.cross_attn_ffn = PositionwiseFFN(d_model, d_ff, dropout)

        if use_spatial:
            self.spatial = AdaptiveChebyshevGraphConv(
                n_nodes, d_model, cheb_k=cheb_k, dropout=dropout,
                restrict_static_graph=restrict_static_graph,
            )
            self.spatial_ffn = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(self, h_target, h_exo, A_static, A_adaptive):
        h_target = self.target_temporal(h_target)
        h_target = self.target_temporal_ffn(h_target)

        if self.use_spatial:
            h_target = self.spatial(h_target, A_static, A_adaptive)
            h_target = self.spatial_ffn(h_target)

        if self.use_exo_attn:
            h_exo = self.exo_attn(h_exo)
            h_exo = self.exo_attn_ffn(h_exo)

        if self.use_cross:
            h_target = self.cross_attn(h_target, h_exo)
            h_target = self.cross_attn_ffn(h_target)

        return h_target, h_exo


class GraphAutoregressivePredictionHead(nn.Module):
    """Autoregressive graph decoder initialized from ExoGNN patch states."""

    def __init__(self, n_patches, d_model, pred_len, dropout=0.1,
                 use_context=True, use_future_context=True):
        super().__init__()
        self.pred_len = pred_len
        self.use_context = use_context
        self.use_future_context = use_future_context

        init_dim = n_patches * d_model + 1
        if use_context:
            init_dim += d_model

        self.init_proj = nn.Sequential(
            nn.LayerNorm(init_dim),
            nn.Linear(init_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.horizon_embedding = nn.Parameter(torch.empty(pred_len, d_model))
        nn.init.normal_(self.horizon_embedding, std=0.02)

        step_dim = 2 + d_model * 2
        if use_context:
            step_dim += d_model
        if use_future_context:
            step_dim += d_model

        self.step_proj = nn.Sequential(
            nn.Linear(step_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gru_cell = nn.GRUCell(d_model, d_model)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )
        self.delta_gate = nn.Parameter(torch.tensor(_scale_to_logit(0.2), dtype=torch.float32))

    def forward(self, h, exo_context=None, future_context=None, last_target=None, A_static=None):
        # h: [B, N, O, P, D], last_target: [B, N, O]
        B, N, O, P, D = h.shape
        target_flat = h.contiguous().view(B, N, O, P * D)

        if last_target is None:
            last_target = h.new_zeros(B, N, O)

        init_pieces = [target_flat, last_target.unsqueeze(-1)]
        if self.use_context:
            if exo_context is None:
                exo_context = h.new_zeros(B, N, D)
            exo = exo_context.unsqueeze(2).expand(-1, -1, O, -1)
            init_pieces.append(exo)
        else:
            exo = None

        hidden = self.init_proj(torch.cat(init_pieces, dim=-1))
        prev = last_target

        if A_static is None:
            A_static = torch.eye(N, device=h.device, dtype=h.dtype)
        A_static = A_static.to(device=h.device, dtype=h.dtype)
        A_static = A_static / A_static.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        preds = []
        for step in range(self.pred_len):
            neighbor_prev = torch.einsum("nm,bmo->bno", A_static, prev)
            neighbor_hidden = torch.einsum("nm,bmod->bnod", A_static, hidden)
            horizon = self.horizon_embedding[step].view(1, 1, 1, D).expand(B, N, O, -1)

            step_pieces = [
                prev.unsqueeze(-1),
                neighbor_prev.unsqueeze(-1),
                neighbor_hidden,
                horizon,
            ]
            if self.use_context:
                step_pieces.append(exo)
            if self.use_future_context:
                if future_context is None:
                    future_step = h.new_zeros(B, N, D)
                else:
                    future_step = future_context[:, :, step, :]
                step_pieces.append(future_step.unsqueeze(2).expand(-1, -1, O, -1))

            step_input = self.step_proj(torch.cat(step_pieces, dim=-1))
            hidden = self.gru_cell(
                step_input.reshape(B * N * O, D),
                hidden.reshape(B * N * O, D),
            ).view(B, N, O, D)

            delta = self.out_proj(hidden).squeeze(-1)
            prev = prev + torch.sigmoid(self.delta_gate) * delta
            preds.append(prev)

        return torch.stack(preds, dim=2)


class ExoGNN(BaseModel):
    """
    Patch-TimeXer-style ExoGNN with Chebyshev graph decoding.

    Input: [B, T, N, F] (batch, time, node, feature)
    Output: [B, H, N, O] (batch, horizon, node, output_dim)
    """

    def __init__(self, adj_mx, d_model=64, n_heads=8, e_layers=2, d_ff=256,
                 dropout=0.1, patch_len=2, patch_stride=2,
                 target_var_indices=None, exogenous_var_indices=None,
                 coords=None,
                 use_spatial=True, use_exo_attn=True, use_cross=True,
                 cheb_k=3, use_future_exo=True,
                 use_exo_head_context=True, use_future_step_context=True,
                 restrict_static_graph=False, **args):
        super(ExoGNN, self).__init__(**args)

        self.cheb_k = max(1, int(cheb_k))
        self.use_spatial = bool(use_spatial)
        self.use_exo_attn = bool(use_exo_attn)
        self.use_cross = bool(use_cross)
        self.use_future_exo = bool(use_future_exo)
        self.use_future_step_context = bool(use_future_step_context)

        n_nodes = self.node_num
        n_vars = self.input_dim
        seq_len = self.seq_len
        pred_len = self.horizon

        target_var_indices = _parse_indices(target_var_indices, default=[0])
        default_exogenous = [idx for idx in range(n_vars) if idx not in set(target_var_indices)]
        exogenous_var_indices = _parse_indices(exogenous_var_indices, default=default_exogenous)

        for idx in target_var_indices:
            if idx < 0 or idx >= n_vars:
                raise ValueError(f"target index {idx} is out of range for input_dim={n_vars}")
        if len(set(target_var_indices)) != len(target_var_indices):
            raise ValueError("target_var_indices contains duplicates")

        for idx in exogenous_var_indices:
            if idx < 0 or idx >= n_vars:
                raise ValueError(f"exogenous index {idx} is out of range for input_dim={n_vars}")

        overlap = set(target_var_indices) & set(exogenous_var_indices)
        if overlap:
            raise ValueError(f"target and exogenous indices overlap: {sorted(overlap)}")

        all_used_indices = sorted(set(exogenous_var_indices) | set(target_var_indices))
        orig_to_local = {orig_idx: local_idx for local_idx, orig_idx in enumerate(all_used_indices)}
        target_var_indices_local = [orig_to_local[i] for i in target_var_indices]
        exogenous_var_indices_local = [orig_to_local[i] for i in exogenous_var_indices]

        self.register_buffer("orig_var_idx", torch.tensor(all_used_indices, dtype=torch.long))

        self.n_vars = len(all_used_indices)
        self.n_vars_orig = n_vars
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_out = len(target_var_indices_local)
        self.n_exo = len(exogenous_var_indices_local)
        self.register_buffer("target_idx", torch.tensor(target_var_indices_local, dtype=torch.long))
        self.register_buffer("exo_idx", torch.tensor(exogenous_var_indices_local, dtype=torch.long))

        use_exo = self.n_exo > 0
        use_exo_attn = use_exo and self.use_exo_attn
        use_cross = use_exo and self.use_cross

        self.target_embedding = TargetPatchEmbedding(
            seq_len=seq_len,
            patch_len=patch_len,
            patch_stride=patch_stride,
            d_model=d_model,
            dropout=dropout,
        )
        self.n_patches = self.target_embedding.n_patches

        self.exo_embedding = ExogenousEmbedding(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            dropout=dropout,
            use_future=self.use_future_exo,
        )

        self.use_spatial_pe = coords is not None
        if self.use_spatial_pe:
            self.spatial_pe = SpatialPositionalEncoding(d_model, coords)

        self.target_variate_pe = VariatePositionalEncoding(self.n_out, d_model)
        self.exo_variate_pe = VariatePositionalEncoding(max(self.n_exo, 1), d_model)

        # DeepAirEngine passes future side information as features 1:.
        future_exo_positions = []
        future_source_indices = []
        for local_pos, orig_idx in enumerate(exogenous_var_indices):
            if orig_idx > 0:
                future_exo_positions.append(local_pos)
                future_source_indices.append(orig_idx - 1)
        self.register_buffer("target_orig_idx", torch.tensor(target_var_indices, dtype=torch.long))
        self.register_buffer("exo_orig_idx", torch.tensor(exogenous_var_indices, dtype=torch.long))
        self.register_buffer("future_exo_positions", torch.tensor(future_exo_positions, dtype=torch.long))
        self.register_buffer("future_source_indices", torch.tensor(future_source_indices, dtype=torch.long))

        self.blocks = nn.ModuleList([
            PatchXerGNNBlock(
                n_nodes, d_model, n_heads, d_ff, dropout,
                use_exo_attn=use_exo_attn,
                use_cross=use_cross,
                use_spatial=self.use_spatial,
                cheb_k=self.cheb_k,
                restrict_static_graph=restrict_static_graph,
            )
            for _ in range(e_layers)
        ])

        if self.use_future_step_context and self.n_exo > 0:
            self.future_step_proj = nn.Sequential(
                nn.Linear(self.n_exo, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )

        self.head = GraphAutoregressivePredictionHead(
            self.n_patches, d_model, pred_len, dropout,
            use_context=use_exo_head_context,
            use_future_context=self.use_future_step_context,
        )

        A_static = self._build_static_adjacency(n_nodes, adj_mx)
        self.register_buffer("A_static", torch.from_numpy(A_static))

        self.adaptive_emb1 = nn.Parameter(torch.empty(n_nodes, d_model))
        self.adaptive_emb2 = nn.Parameter(torch.empty(n_nodes, d_model))
        nn.init.xavier_uniform_(self.adaptive_emb1)
        nn.init.xavier_uniform_(self.adaptive_emb2)

    @staticmethod
    def _build_static_adjacency(n_nodes, adj_mx):
        if adj_mx is not None:
            if isinstance(adj_mx, torch.Tensor):
                A = adj_mx.cpu().numpy().astype(np.float32)
            else:
                A = np.array(adj_mx, dtype=np.float32)
            if A.shape != (n_nodes, n_nodes):
                raise ValueError(f"adj_matrix shape {A.shape} != ({n_nodes},{n_nodes})")
        else:
            A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        A = np.maximum(A, 0.0)
        A = A + np.eye(n_nodes, dtype=np.float32)
        row_sum = A.sum(axis=-1, keepdims=True)
        A = A / np.maximum(row_sum, 1e-6)
        return A.astype(np.float32)

    def _get_adaptive_adjacency(self):
        A_raw = F.relu(self.adaptive_emb1 @ self.adaptive_emb2.T)
        A_adaptive = F.softmax(A_raw, dim=-1)
        I = torch.eye(
            A_adaptive.shape[0], device=A_adaptive.device, dtype=A_adaptive.dtype
        )
        A_adaptive = A_adaptive + I
        return A_adaptive / A_adaptive.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _select_future_exo(self, label, B, N):
        if (
            label is None
            or not self.use_future_exo
            or self.n_exo == 0
            or self.future_source_indices.numel() == 0
        ):
            return None

        if label.shape[-1] == self.n_vars_orig - 1:
            future = label.index_select(dim=-1, index=self.future_source_indices)
        elif label.shape[-1] == self.n_vars_orig:
            future = label.index_select(dim=-1, index=self.exo_orig_idx[self.future_exo_positions])
        else:
            return None

        future = future.permute(0, 2, 3, 1).contiguous()
        if future.shape[2] == self.n_exo:
            return future

        aligned = future.new_zeros(B, N, self.n_exo, self.horizon)
        aligned.index_copy_(2, self.future_exo_positions, future)
        return aligned

    def _get_future_step_context(self, future_exo, B, N):
        if (
            not self.use_future_step_context
            or future_exo is None
            or self.n_exo == 0
            or not hasattr(self, "future_step_proj")
        ):
            return None
        future_steps = future_exo.permute(0, 1, 3, 2).contiguous()
        return self.future_step_proj(future_steps)

    def forward(self, input, label=None):
        """
        input: [B, T, N, F] (CauAir format)
        returns: [B, H, N, O]
        """
        x = input.permute(0, 2, 3, 1)  # [B, N, F, T]

        x = x.index_select(dim=2, index=self.orig_var_idx)

        x_target = x.index_select(dim=2, index=self.target_idx)
        x_exo = x.index_select(dim=2, index=self.exo_idx)
        future_exo = self._select_future_exo(label, input.shape[0], input.shape[2])
        future_step_context = self._get_future_step_context(future_exo, input.shape[0], input.shape[2])

        h_target = self.target_embedding(x_target)
        h_exo = self.exo_embedding(x_exo, future_exo)

        B, N, O, P, D = h_target.shape
        E = h_exo.shape[2]

        if self.use_spatial_pe:
            spatial_pe = self.spatial_pe(N)
            h_target = h_target + spatial_pe.view(1, N, 1, 1, D)
            if E > 0:
                h_exo = h_exo + spatial_pe.view(1, N, 1, D)

        target_ce = self.target_variate_pe()
        h_target = h_target + target_ce.view(1, 1, O, 1, D)

        if E > 0:
            exo_ce = self.exo_variate_pe()
            h_exo = h_exo + exo_ce[:E].view(1, 1, E, D)

        A_static = self.A_static
        A_adaptive = self._get_adaptive_adjacency() if self.use_spatial else None

        for block in self.blocks:
            h_target, h_exo = block(h_target, h_exo, A_static, A_adaptive)

        exo_context = h_exo.mean(dim=2) if h_exo.shape[2] > 0 else None
        last_target = input[:, -1].index_select(dim=-1, index=self.target_orig_idx)
        out = self.head(h_target, exo_context, future_step_context, last_target, A_static)

        out = out.permute(0, 2, 1, 3)  # [B, H, N, O]
        return out

    def get_graph_info(self):
        spatial_support_gates = []

        for block in self.blocks:
            if not (hasattr(block, "spatial") and block.use_spatial):
                continue
            spatial = block.spatial
            if hasattr(spatial, "get_support_gates"):
                spatial_support_gates.append(spatial.get_support_gates())

        return {
            "alpha_values": [g["residual_gate"] for g in spatial_support_gates],
            "sparse_attention_weights": [],
            "spatial_support_gates": spatial_support_gates,
            "A_static": self.A_static.detach(),
            "target_idx": self.target_idx.detach().cpu().tolist(),
            "exo_idx": self.exo_idx.detach().cpu().tolist(),
            "orig_var_idx": self.orig_var_idx.detach().cpu().tolist(),
            "n_vars": self.n_vars,
            "n_vars_orig": self.n_vars_orig,
            "patch_len": self.patch_len,
            "patch_stride": self.patch_stride,
            "n_patches": self.n_patches,
            "use_spatial_pe": self.use_spatial_pe,
            "use_spatial": self.use_spatial,
            "use_exo_attn": self.use_exo_attn,
            "use_cross": self.use_cross,
            "cheb_k": self.cheb_k,
            "use_future_exo": self.use_future_exo,
            "use_future_step_context": self.use_future_step_context,
            "spatial_layer": "chebyshev",
        }


FG_ExoGNN = ExoGNN
