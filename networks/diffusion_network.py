"""
This implementation follows the original DiffusionNet (Sharp et al.) and DPFM (Attaki et al.) implementations.
CAUTION: THIS IS DIFFERENT FROM IMPLEMENTATIONS of (Cao et al. 2022-2024 works and its derivatives eg. Bastian&Xie et al. 2024, etc.)
The single line of core difference is not having the complex spatial gradient feature computation bug. 
layer names are kept different to avoid confusion between implementations and a hard wall to prevent ckpt loading confusion.
In a rule of thumb, this implementation should be favored for correct computation and for future research
However in practice the actual effect has been empirically minimal and the exact impact is to be studied.
"""

import torch
import torch.nn as nn
from utils.registry import NETWORK_REGISTRY
from utils.geometry_util import compute_wks_autoscale, data_augmentation, hash_arrays, torch2np
from utils.temp_seed_util import temp_seed
def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        return utils.cmatmul(utils.ensure_complex(basis), utils.ensure_complex(values)) # this is not defined even in the original DPFM code, so it's fine
    else:
        return torch.matmul(basis, values)

class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.
    In the spectral domain this becomes
        f_out = e ^ (lambda_i t) f_in
    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal
      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values
    """

    def __init__(self, C_inout, method="spectral"):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method  # one of ['spectral', 'implicit_dense']

        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout
                )
            )

        if self.method == "spectral":

            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            # Diffuse
            time = self.diffusion_time
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec

            # Transform back to per-vertex
            x_diffuse = from_basis(x_diffuse_spec, evecs)

        elif self.method == "implicit_dense":
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)

            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")

        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.

    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors  # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vectorsBreal = self.A(vectors[..., 0])
            vectorsBimag = self.A(vectors[..., 1])

        dots = vectorsA[..., 0] * vectorsBreal + vectorsA[..., 1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes.
    """

    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = i + 2 == len(layer_sizes)

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i), nn.Dropout(p=0.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(name + "_mlp_act_{:03d}".format(i), activation())


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(
        self,
        C_width,
        mlp_hidden_dims,
        dropout=True,
        diffusion_method="spectral",
        with_gradient_features=True,
        with_gradient_rotations=True,
    ):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)

        self.MLP_C = 2 * self.C_width

        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                self.C_width, with_gradient_rotations=self.with_gradient_rotations
            )
            self.MLP_C += self.C_width

        # MLPs
        self.mlp = MiniMLP(
            [self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout
        )

    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0]  # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width
                )
            )

        # Diffusion block
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = (
                []
            )  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b, ...], x_diffuse[b, ...])
                x_gradY = torch.mm(gradY[b, ...], x_diffuse[b, ...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad)

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out

@NETWORK_REGISTRY.register()
class DiffusionNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=128,
        n_block=4,
        last_activation=None,
        outputs_at="vertices",
        mlp_hidden_dims=None,
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=True,
        diffusion_method="spectral",
        input_type="xyz",
        augmentation={'train': {}, 'test': {}}
    ):
        """
        Construct a DiffusionNet.
        Parameters:
            in_channels (int):                     input dimension
            out_channels (int):                    output dimension
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces'].
            (default 'vertices', aka points for a point cloud)
            hidden_channels (int):                  dimension of internal DiffusionNet blocks (default: 128)
            n_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [hidden_channels, hidden_channels])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0,
            saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient.
            Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DiffusionNet, self).__init__()

        # Store parameters
        self.input_type = input_type
        # augmentation
        self.DEFAULT_TRAIN_AUGMENTATIONS = {
            'rot_x': 30.0,
            'rot_y': 30.0,
            'rot_z': 30.0,
            'std': 0.01,
            'noise_clip': 0.05,
            'scale_min': 0.9,
            'scale_max': 1.1
        }
        self.DEFAULT_TEST_AUGMENTATIONS = {
            'rot_x': 0.0,
            'rot_y': 0.0,
            'rot_z': 0.0,
            'std': 0.0,
            'noise_clip': 0.0,
            'scale_min': 1.0,
            'scale_max': 1.0
        }
        self.train_augmentation = {**self.DEFAULT_TRAIN_AUGMENTATIONS, **(augmentation.get("train", {}) or {})}
        self.test_augmentation = {**self.DEFAULT_TEST_AUGMENTATIONS, **(augmentation.get("test", {}) or {})}

        if self.input_type == 'xyz':
            print("Settings:")
            print(f"  Input type: {self.input_type}")
            print(f"  Train augmentations: {self.train_augmentation}")
            print(f"  Test  augmentations: {self.test_augmentation}")

        # Basic parameters
        self.C_in = in_channels
        self.C_out = out_channels
        self.C_width = hidden_channels
        self.N_block = n_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ["vertices", "edges", "faces"]:
            raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [self.C_width, self.C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ["spectral", "implicit_dense"]:
            raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # # Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(self.C_in, self.C_width)
        self.last_lin = nn.Linear(self.C_width, self.C_out)

        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(
                C_width=self.C_width,
                mlp_hidden_dims=self.mlp_hidden_dims,
                dropout=self.dropout,
                diffusion_method=self.diffusion_method,
                with_gradient_features=self.with_gradient_features,
                with_gradient_rotations=self.with_gradient_rotations,
            )

            self.blocks.append(block)
            self.add_module("block_" + str(i_block), self.blocks[-1])

    def forward(
        self,
        data
    ):
        """
        A forward pass on the DiffusionNet.
        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].
        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet,
        not all are strictly necessary.
        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """
        if self.input_type == 'xyz':
            x_in = data['xyz']
            if self.training:
                x_in = data_augmentation(x_in, **self.train_augmentation)
            else:
                hash = hash_arrays([torch2np(data['verts']), torch2np(data['faces'])])
                seed = int(hash, 16) % (2**32)
                with temp_seed(seed):
                    x_in = data_augmentation(x_in, **self.test_augmentation)
        elif self.input_type == 'wks':
            x_in = data['wks']
        elif self.input_type == 'hks':
            x_in = data['hks']
        elif self.input_type == 'dino':
            x_in = data['dino']
        else:
            x_in = data[self.input_type]
        
        mass, L, evals, evecs, gradX, gradY = data['operators']['mass'], data['operators']['L'], \
                                                    data['operators']['evals'], data['operators']['evecs'], \
                                                    data['operators']['gradX'], data['operators']['gradY']

        # Apply the first linear layer
        x = self.first_lin(x_in)

        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)

        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == "vertices":
            x_out = x

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        return x_out
