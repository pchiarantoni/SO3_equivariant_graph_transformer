# SO(3) Equivariant transformer with graph attention based on Equiformer https://arxiv.org/pdf/2206.11990
# test with
#    python3 standalone_Equiformer_w_batch.py

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from e3nn.nn import Gate
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)



def scatter_sum(src, index, dim_size=None):
    """
    PyTorch implementation of scatter sum along dim=0. torch_scatter not working on ICMS

    Args:
        src (Tensor): values to scatter, shape (num_edges, F)
        index (LongTensor): target indices for each row of src, shape (num_edges,)
        dim_size (int, optional): number of output rows (usually number of nodes)

    Returns:
        out (Tensor): aggregated output, shape (dim_size, F)
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def radius_graph_torch(pos, r, loop=False):
    """
    Build a radius graph without torch_cluster.

    Args:
        pos (Tensor): [N, 3] node positions
        r (float): cutoff radius
        loop (bool): whether to include self-loops

    Returns:
        edge_index (Tensor): [2, num_edges]
    """
    N = pos.size(0)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
    dist2 = (diff ** 2).sum(dim=-1)             # (N, N)
    mask = dist2 <= r**2
    if not loop:
        mask.fill_diagonal_(False)
    edge_src, edge_dst = torch.nonzero(mask, as_tuple=True)
    return torch.stack([edge_src, edge_dst], dim=0)



class EquivariantLayerNormIrreps(nn.Module):
    def __init__(self, irreps: o3.Irreps, eps=1e-6):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.gammas = nn.ParameterDict()
        self.shifts = nn.ParameterDict()  # <-- learnable shift for l=0

        for i, (mul, ir) in enumerate(self.irreps):
            self.gammas[str(i)] = nn.Parameter(torch.ones(mul))
            if ir.l == 0:
                # add a learnable shift per scalar multiplicity
                self.shifts[str(i)] = nn.Parameter(torch.zeros(mul))

    def forward(self, x):
        out = []
        start = 0
        for i, (mul, ir) in enumerate(self.irreps):
            dim = mul * (2 * ir.l + 1)
            x_chunk = x[:, start:start + dim].view(x.shape[0], mul, 2*ir.l+1)

            norm = torch.linalg.norm(x_chunk, dim=-1, keepdim=True)
            rms = torch.sqrt((norm ** 2).mean(dim=(0,1), keepdim=True) + self.eps)
            x_normed = x_chunk / rms
            x_normed = x_normed * self.gammas[str(i)].view(1, mul, 1)

            # add learnable shift for scalars
            if ir.l == 0:
                x_normed = x_normed + self.shifts[str(i)].view(1, mul, 1)

            out.append(x_normed.view(x.shape[0], -1))
            start += dim

        return torch.cat(out, dim=1)



class SO3TransformerLayer(nn.Module):
    def __init__(
        self,
        irreps_node,
        irreps_edge,
        max_radius,
        number_of_basis,
        lmax_sh=1
    ):
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        # Edge spherical harmonics
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax_sh)

        # edge length embedding parameters
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis

        # Linear node features embedding
        self.node_feats_emb = o3.Linear(self.irreps_node, self.irreps_node)

        # Depthwise Tensor Product
        self.DTP = FullyConnectedTensorProduct(self.irreps_node, self.irreps_sh, self.irreps_edge, shared_weights=False)
        self.DTP_edge = FullyConnectedTensorProduct(self.irreps_edge, self.irreps_sh, self.irreps_edge, shared_weights=False)

        # conditioning weights for DTP
        self.weights_net = FullyConnectedNet([self.number_of_basis, 16, 16, self.DTP.weight_numel], act=torch.nn.functional.silu)
        #self.weights_net_nlmp = FullyConnectedNet([1, 16, self.DTP_edge.weight_numel], act=torch.nn.functional.silu)

        self.weights_nlmp = nn.Parameter(torch.rand(self.DTP_edge.weight_numel))  # shape: [weight_numel]

        # Linear edge features embedding
        self.edge_feats_emb = o3.Linear(self.irreps_edge, self.irreps_edge)

        # Linear edge to node
        self.edge_to_node = o3.Linear(self.irreps_edge, self.irreps_node)

        # Linear Non-Linear message passing
        self.lin_nlmp = o3.Linear(self.irreps_edge, self.irreps_edge)

        # Learnable tensor of scalars for attention
        num_scalars_edge = sum(mul for mul, ir in self.irreps_edge if ir.l == 0)
        self.a = nn.Parameter(torch.rand(num_scalars_edge))


        # ---------- prepare GATING EDGES partition ----------
        # scalar outputs (0e) that will pass through scalar activations
        irreps_scalars_edge = self.irreps_edge.filter("0e")
        # gated irreps: non-scalars that we want to gate (e.g. 1o, 2e, ...)
        irreps_gated_edge = o3.Irreps([mul_ir for mul_ir in self.irreps_edge if mul_ir not in irreps_scalars_edge])
        # irreps_gates: for each *irreducible* in irreps_gated we need one scalar gate.
        # The Gate constructor requires irreps_gates.num_irreps == irreps_gated.num_irreps
        # So we create that many scalar (0e) irreps.
        irreps_gates_edge = o3.Irreps(f"{irreps_gated_edge.num_irreps}x0e") if irreps_gated_edge.num_irreps > 0 else o3.Irreps("0x0e")
        # The Gate expects the input ordering: scalars + gates + gated
        irreps_gate_edge_full = irreps_scalars_edge + irreps_gates_edge + irreps_gated_edge
        
        self.edge_to_gating = o3.Linear(self.irreps_edge, irreps_gate_edge_full)  #Linear should work fine because it is only adding scalars. NOTE: if higher order tensors are need from a given pair of irreps, FCTP must be used 
        
        # ---------- Gate construction ----------
        # Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
        # activations are lists; match lengths or pass single callable (e3nn will broadcast)
        act_scalars_edge = [torch.nn.functional.silu] if len(irreps_scalars_edge) > 0 else []
        act_gates_edge = [torch.tanh] if len(irreps_gates_edge) > 0 else []

        # create Gate edge module
        self.gate_edge = Gate(
            irreps_scalars_edge, act_scalars_edge,
            irreps_gates_edge, act_gates_edge,
            irreps_gated_edge
        )


        # ---------- prepare GATING NODES partition ----------
        irreps_scalars_node = self.irreps_node.filter("0e")
        irreps_gated_node = o3.Irreps([mul_ir for mul_ir in self.irreps_node if mul_ir not in irreps_scalars_node])
        irreps_gates_node = o3.Irreps(f"{irreps_gated_node.num_irreps}x0e") if irreps_gated_node.num_irreps > 0 else o3.Irreps("0x0e")
        self.irreps_gate_node_full = irreps_scalars_node + irreps_gates_node + irreps_gated_node
        
        self.node_to_gating = o3.Linear(self.irreps_node, self.irreps_gate_node_full) 

        act_scalars_node = [torch.nn.functional.silu] if len(irreps_scalars_node) > 0 else []
        act_gates_node = [torch.tanh] if len(irreps_gates_node) > 0 else []

        self.gate_node = Gate(
            irreps_scalars_node, act_scalars_node,
            irreps_gates_node, act_gates_node,
            irreps_gated_node
        )

        # LayerNorm
        self.ln1 = EquivariantLayerNormIrreps(self.irreps_node)
        self.ln2 = EquivariantLayerNormIrreps(self.irreps_node)

        # Gated feedforward
        self.gated_ff = nn.Sequential(
            o3.Linear(self.irreps_node, self.irreps_gate_node_full),
            self.gate_node,
            o3.Linear(self.irreps_node, self.irreps_node)
        )
 

    def forward(self, f, pos, edge_index):
        

#        print("f", f)
        # Pre-attention LayerNorm
        f_ln1 = self.ln1(f)
#        print("f_ln1", f_ln1)
        # --- EQUIVARIANT GRAPH ATTENTION ---
        edge_src, edge_dst = edge_index

        # Build edge features
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)

        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded * (self.number_of_basis ** 0.5)
        
#        print("edge_length_emb", edge_length_embedded)

        # Create edge features from node features
        edge_feats = self.node_feats_emb(f[edge_src]) + self.node_feats_emb(f[edge_dst])    # NOTE irreps = irreps_node

#        print("edge feats", edge_feats)

        # Edge spherical harmonics
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')

#        print("edge_sh", edge_sh)

        # First DTP
        f_ij = self.edge_feats_emb(self.DTP(edge_feats, edge_sh, weight=self.weights_net(edge_length_embedded))) #EQ.3   NOTE: this DTP can be conditioned on scalar observables
#        print("f_ij", f_ij)
        # Split edge features in scalar/non_scalar
        scalar_f_ij = []

        start = 0
        for mul, ir in self.irreps_edge:  # iterate over irreps
            dim = mul * (2 * ir.l + 1)
            chunk = f_ij[:, start:start+dim]  # select slice

            if ir.l == 0:  # scalar
                scalar_f_ij.append(chunk)

            start += dim

        # concatenate all selected chunks along last dim
        scalar_f_ij = torch.cat(scalar_f_ij, dim=1) if scalar_f_ij else None
#        print("scalar f_ij", scalar_f_ij)

        # Scalar attention weights (EQ.4)
        z_ij = (F.leaky_relu(scalar_f_ij, negative_slope=0.01) * self.a).sum(dim=1)
        z_ij = torch.clamp(z_ij, min=-10, max=10)
        exp_ij = z_ij.exp().unsqueeze(1) 
        z = scatter_sum(exp_ij, edge_dst, dim_size=len(f_ln1))
        z[z == 0] = 1
        a_ij = exp_ij / z[edge_dst] 
#        print("a_ij", a_ij)        

        # Non-linear message passing (EQ.5)
        f_ij_to_gate = self.edge_to_gating(f_ij)
        mu_ij = self.gate_edge(f_ij_to_gate)        

#        print("mu_ij", mu_ij)

        weights = self.weights_nlmp.unsqueeze(0)

        v_ij = self.lin_nlmp(self.DTP_edge(mu_ij, edge_sh, weight=weights)) #NOTE weights should be learnable and sample independent 
#        print("v_ij", v_ij)

        # Aggregate messages -> out has shape [N, irreps_node.dim]
        out_attn = self.edge_to_node(scatter_sum(a_ij * v_ij, edge_dst, dim_size=len(f_ln1)))

#        print("out_attn", out_attn)

        # First residual connection
        f_res1 = f + out_attn 

#        print("f_res1", f_res1)

        # Second LayerNorm
        f_ln2 = self.ln2(f_res1)

#        print("f_ln2", f_ln2)

        # Apply gated feedforwnard and second residual connection to the full feature vector
        out = f_res1 + self.gated_ff(f_ln2)

#        print("gated ff", self.gated_ff(f_ln2))

        return out

class SO3TransformerStack(nn.Module):
    """
    Stack several SO3TransformerLayer in series.
    Applies final Equivariant LayerNorm + Gated FeedForward at the end.
    """
    def __init__(
        self,
        num_layers,
        irreps_node,
        irreps_edge,
        max_radius,
        number_of_basis,
        lmax_sh=3
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.irreps_node = o3.Irreps(irreps_node)

        # First layer input is irreps_input
        for i in range(num_layers):
            layer = SO3TransformerLayer(
                irreps_node=irreps_node,
                irreps_edge=irreps_edge,
                max_radius=max_radius,
                number_of_basis=number_of_basis,
                lmax_sh=lmax_sh
            )
            self.layers.append(layer)

        # Final equivariant layer norm and gated feedforward
        self.final_ln = EquivariantLayerNormIrreps(o3.Irreps(irreps_node))

        # ---------- prepare gating partition ----------
        irreps_scalars = self.irreps_node.filter("0e")
        irreps_gated = o3.Irreps([mul_ir for mul_ir in self.irreps_node if mul_ir not in irreps_scalars])
        irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e") if irreps_gated.num_irreps > 0 else o3.Irreps("0x0e")

        irreps_out_full = irreps_scalars + irreps_gates + irreps_gated

        act_scalars = [torch.nn.functional.silu] if len(irreps_scalars) > 0 else []
        act_gates = [torch.tanh] if len(irreps_gates) > 0 else []

        # store for forward bookkeeping
        self.irreps_out_full = irreps_out_full
        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated

        # create Gate module
        self.gate = Gate(
            irreps_scalars, act_scalars,
            irreps_gates, act_gates,
            irreps_gated
        )

        # Gated feedforward
        self.final_gated_ff = nn.Sequential(
            o3.Linear(self.irreps_node, self.irreps_out_full),
            self.gate,
            o3.Linear(self.irreps_node, self.irreps_node)
        )

    def forward(self, f, pos, edge_index):
        out = f
        for layer in self.layers:
            out = layer(out, pos, edge_index)
        out = self.final_ln(out)
        out = self.final_gated_ff(out)
        return out

class MultiHeadSO3Transformer(nn.Module):
    """
    Wrapper around SO3TransformerStack that supports multiple attention heads.
    Each head gets the same input features, outputs are concatenated, then projected to output irreps.
    """
    def __init__(
        self,
        num_heads,
        stack_params,  # dict containing all params needed for SO3TransformerStack
    ):
        super().__init__()
        self.num_heads = num_heads
        
        # Create a separate stack for each head
        self.stacks = nn.ModuleList([
            SO3TransformerStack(**stack_params) for _ in range(num_heads)
        ])

        # Final linear to mix concatenated head outputs back to output irreps
        irreps_node = o3.Irreps(stack_params['irreps_node'])
        self.mix = o3.Linear(irreps_node * num_heads, irreps_node)

    def forward(self, f, pos, edge_index):
        # Apply each head
        head_outputs = [stack(f, pos, edge_index) for stack in self.stacks]

        # Concatenate along feature dimension
        concat_out = torch.cat(head_outputs, dim=1)

        # Project back to desired output irreps
        out = self.mix(concat_out)
        return out

class BatchedSO3Transformer(nn.Module):
    """
    Wrapper to handle batches of graphs for SO3Transformer.
    Each graph has node features f_i, positions pos_i, and edge_index_i.
    """
    def __init__(self, transformer_model):
        super().__init__()
        self.model = transformer_model

    def forward(self, f_list, pos_list, edge_index_list):
        """
        Args:
            f_list: list of [N_i, F] node feature tensors
            pos_list: list of [N_i, 3] position tensors
            edge_index_list: list of [2, E_i] edge index tensors

        Returns:
            out_list: list of [N_i, F_out] output tensors per graph
        """
        # Total number of nodes
        cum_nodes = 0
        f_batch = []
        pos_batch = []
        edge_index_batch = []

        for f, pos, edge_index in zip(f_list, pos_list, edge_index_list):
            N = f.shape[0]
            f_batch.append(f)
            pos_batch.append(pos)

            # Offset edge indices by cumulative number of nodes so far
            edge_index_batch.append(edge_index + cum_nodes)
            cum_nodes += N

        # Concatenate all graphs into a single batch
        f_batch = torch.cat(f_batch, dim=0)
        pos_batch = torch.cat(pos_batch, dim=0)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)

        # Forward through transformer
        out_batch = self.model(f_batch, pos_batch, edge_index_batch)

        # Split outputs per graph
        out_list = []
        start = 0
        for f in f_list:
            N = f.shape[0]
            out_list.append(out_batch[start:start+N])
            start += N

        return out_list

def main():
    #torch.manual_seed(11)

    # -----------------------
    # Define irreps
    # -----------------------
    irreps_node  = "1x0e + 1x1o + 1x2o + 1x2e"
    irreps_edge  = "1x0e + 1x1o + 1x2o + 1x2e"
    print("irreps node", irreps_node)
    print("irreps edge", irreps_edge)

    # -----------------------
    # Stack parameters
    # -----------------------
    stack_params = dict(
        num_layers=1,
        irreps_node=irreps_node,
        irreps_edge=irreps_edge,
        max_radius=2.0,
        number_of_basis=10,
        lmax_sh=2
    )

    # -----------------------
    # Build multihead transformer
    # -----------------------
    model = MultiHeadSO3Transformer(
        num_heads=1,
        stack_params=stack_params
    )

    # -----------------------
    # Create a batch of 3 random graphs
    # -----------------------
    num_graphs = 1
    N_nodes = [200, 150, 250]  # number of nodes per graph

    f_list = []
    pos_list = []
    edge_list = []

    irreps_node = o3.Irreps(irreps_node)

    for N in N_nodes:
        # Positions
        pos = torch.randn(N, 3)

        pos_list.append(pos)

        # Node features
        f_chunks = []
        for mul, ir in irreps_node:
            if ir.l == 0:
                f_chunks.append(torch.randn(N, mul))
            else:
                f_chunks.append(torch.randn(N, mul * (2*ir.l+1)))
        f = torch.cat(f_chunks, dim=1)
        f_list.append(f)

        # Edges
        edges = radius_graph_torch(pos, r=2.5)
        edge_list.append(edges)

    # -----------------------
    # Wrap in batch transformer
    # -----------------------
    batched_model = BatchedSO3Transformer(model)

    # Forward pass
    outputs = batched_model(f_list, pos_list, edge_list)

    # Print output shapes per graph
    output_irreps = o3.Irreps(irreps_node)
    print("Output shapes per graph:")
    for i, out in enumerate(outputs):
        print(f"Graph {i}: {out.shape} (should be [N_i, {output_irreps.dim}])")

    # -----------------------
    # Apply a random rotation to all graphs
    # -----------------------
    R = o3.rand_matrix()
    alpha, beta, gamma = o3.matrix_to_angles(R)
    D2 = o3.wigner_D(2, alpha, beta, gamma)  # 5x5 rotation matrix for l=2 irreps

    f_rot_list = []
    pos_rot_list = []
    edge_rot_list = []

    for f, pos in zip(f_list, pos_list):
        # Rotate positions
        pos_rot = pos @ R.T
        pos_rot_list.append(pos_rot)

        # Rotate l=1 (vectors) and l=2 (rank-2) features in f
        f_rot_chunks = []
        start = 0
        for mul, ir in irreps_node:
            dim = mul * (2*ir.l + 1)
            x_chunk = f[:, start:start+dim]

            if ir.l == 1:
                # rotate as Cartesian vectors
                x_chunk = x_chunk.reshape(f.shape[0], mul, 3)
                x_chunk = torch.einsum("ij,nkj->nki", R, x_chunk)
                x_chunk = x_chunk.reshape(f.shape[0], -1)

            elif ir.l == 2:
                # rotate using Wigner-D for l = 2
                x_chunk = x_chunk.reshape(f.shape[0], mul, 5)
                # apply D2 on the last dimension
                x_chunk = torch.einsum("ij,nkj->nki", D2, x_chunk)
                x_chunk = x_chunk.reshape(f.shape[0], -1)

            f_rot_chunks.append(x_chunk)
            start += dim

        f_rot_list.append(torch.cat(f_rot_chunks, dim=1))

        # Recompute edges
        edges_rot = radius_graph_torch(pos_rot, r=2.5)
        edge_rot_list.append(edges_rot)


    # Forward pass on rotated batch
    outputs_rot = batched_model(f_rot_list, pos_rot_list, edge_rot_list)

    # -----------------------
    # Equivariance check per graph
    # -----------------------
    for g_idx, (out_orig, out_rot) in enumerate(zip(outputs, outputs_rot)):
        print(f"\nEquivariance check for Graph {g_idx}:")
        start = 0
        for idx, (mul, ir) in enumerate(irreps_node):
            dim = mul * (2*ir.l + 1)
            if ir.l == 1:  # only vector features rotate
                for v in range(mul):
                    v_orig = out_orig[:, start + v*3:start + (v+1)*3]
                    v_rot  = out_rot[:, start + v*3:start + (v+1)*3]
                    # Rotate original output to compare
                    v_orig_rotated = v_orig @ R.T
                    diff = (v_rot - v_orig_rotated).abs().max().item()
                    print(f"  Max difference for vector irrep #{idx}, vector #{v}: {diff:.3e}")
        
            elif ir.l == 2:  # rank-2 features
                # Wigner D matrix for l=2
                alpha, beta, gamma = o3.matrix_to_angles(R)
                D2 = o3.wigner_D(ir.l, alpha, beta, gamma)
                for v in range(mul):
                    v_orig = out_orig[:, start + v*5:start + (v+1)*5]  # 5 components for l=2
                    v_rot  = out_rot[:, start + v*5: start + (v+1)*5]
                    # Apply Wigner D rotation
                    v_orig_rotated = v_orig @ D2.T  # rotate original
                    diff = (v_rot - v_orig_rotated).abs().max().item()
                    print(f"  Max diff l=2 irrep #{idx}, vec #{v}: {diff:.3e}")

            start += dim


if __name__ == "__main__":
    main()
