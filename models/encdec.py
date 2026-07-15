import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # ic(x.shape)
    x = x.view(
        B, H // window_size, window_size,
        W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size,
        W // window_size, window_size,
        window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module
    with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]):
            The height and width of the window.
        num_heads (int):
            Number of attention heads.
        qkv_bias (bool, optional):
            If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional):
            Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional):
            Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional):
            Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index
        # for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(
            "relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of
                (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, \
            window_size={self.window_size}, \
                num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]):
            Input resulotion.
        num_heads (int):
            Number of attention heads.
        window_size (int):
            Window size.
        shift_size (int):
            Shift size for SW-MSA.
        mlp_ratio (float):
            Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional):
            If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional):
            Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional):
            Dropout rate.
            Default: 0.0
        attn_drop (float, optional):
            Attention dropout rate.
            Default: 0.0
        drop_path (float, optional):
            Stochastic depth rate.
            Default: 0.0
        act_layer (nn.Module, optional):
            Activation layer.
            Default: nn.GELU
        norm_layer (nn.Module, optional):
            Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution,
            # we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        # Round up to a multiple of window_size so window_partition is valid even
        # when the (patch) resolution isn't divisible by window_size (e.g. the
        # 2294-wide RTMA grid). The real forward always passes an already-padded
        # x_size, so this only affects the precomputed mask built at init.
        H = ((H + self.window_size - 1) // self.window_size) * self.window_size
        W = ((W + self.window_size - 1) // self.window_size) * self.window_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing
        # on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(
                x_windows, mask=self.calculate_mask(x_size).to(x.device)
            )

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, \
                input_resolution={self.input_resolution}, \
                    num_heads={self.num_heads}, "
            f"window_size={self.window_size}, \
                shift_size={self.shift_size}, \
                    mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]):
            Resolution of input feature.
        dim (int):
            Number of input channels.
        norm_layer (nn.Module, optional):
            Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int):
            Number of attention heads.
        window_size (int):
            Local window size.
        mlp_ratio (float):
            Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional):
            If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional):
            Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional):
            Dropout rate. Default: 0.0
        attn_drop (float, optional):
            Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional):
            Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional):
            Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional):
            Downsample layer at the end of the layer.
            Default: None
        use_checkpoint (bool):
            Whether to use checkpointing to save memory.
            Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(
                            drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, \
            input_resolution={self.input_resolution}, \
                depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int):
            Number of blocks.
        num_heads (int):
            Number of attention heads.
        window_size (int):
            Local window size.
        mlp_ratio (float):
            Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional):
            If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional):
            Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional):
            Dropout rate.
            Default: 0.0
        attn_drop (float, optional):
            Attention dropout rate.
            Default: 0.0
        drop_path (float | tuple[float], optional):
            Stochastic depth rate.
            Default: 0.0
        norm_layer (nn.Module, optional):
            Normalization layer.
            Default: nn.LayerNorm
        downsample (nn.Module | None, optional):
            Downsample layer at the end of the layer.
            Default: None
        use_checkpoint (bool):
            Whether to use checkpointing to save memory.
            Default: False.
        img_size:
            Input image size.
        patch_size:
            Patch size.
        resi_connection:
            The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
    ):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x, x_size):
        return (
            self.patch_embed(
                self.conv(self.patch_unembed(
                    self.residual_group(x, x_size),
                    x_size))
            )
            + x
        )

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int):
            Image size.
            Default: 224.
        patch_size (int):
            Patch token size.
            Default: 4.
        in_chans (int):
            Number of input image channels.
            Default: 3.
        embed_dim (int):
            Number of linear projection output channels.
            Default: 96.
        norm_layer (nn.Module, optional):
            Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4,
        in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding

    Args:
        img_size (int):
            Image size.
            Default: 224.
        patch_size (int):
            Patch token size.
            Default: 4.
        in_chans (int):
            Number of input image channels.
            Default: 3.
        embed_dim (int):
            Number of linear projection output channels.
            Default: 96.
        norm_layer (nn.Module, optional):
            Normalization layer.
            Default: None
    """

    def __init__(
        self, img_size=224,
        patch_size=4, in_chans=3,
        embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        # B Ph*Pw C
        x = x.transpose(1, 2).view(
            B, self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. " +
                "Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module
        (the difference with Upsample is that it
            always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int):
            Scale factor. Supported scales: 2^n and 3.
        num_feat (int):
            Channel number of intermediate features.

    """

    def __init__(
        self, scale, num_feat, num_out_ch, input_resolution=None
    ):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(
            num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class EncDec(nn.Module):
    r"""EncDec

    Args:
        img_size (int | tuple(int)):
            Input image size.
            Default 64
        patch_size (int | tuple(int)):
            Patch size.
            Default: 1
        in_chans (int):
            Number of input image channels.
            Default: 3
        embed_dim (int):
            Patch embedding dimension.
            Default: 96
        depths (tuple(int)):
            Depth of each Swin Transformer layer.
        num_heads (tuple(int)):
            Number of attention heads in different layers.
        window_size (int):
            Window size.
            Default: 7
        mlp_ratio (float):
            Ratio of mlp hidden dim to embedding dim.
            Default: 4
        qkv_bias (bool):
            If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float):
            Override default qk scale of head_dim ** -0.5 if set.
            Default: None
        drop_rate (float):
            Dropout rate. Default: 0
        attn_drop_rate (float):
            Attention dropout rate. Default: 0
        drop_path_rate (float):
            Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module):
            Normalization layer.
            Default: nn.LayerNorm.
        ape (bool):
            If True,
            add absolute position embedding to the patch embedding.
            Default: False
        patch_norm (bool):
            If True, add normalization after patch embedding.
            Default: True
        use_checkpoint (bool):
            Whether to use checkpointing to save memory.
            Default: False
        upscale:
            Upscale factor.
            2/3/4/8 for image SR,
            1 for denoising and compress artifact reduction
        img_range:
            Image range. 1. or 255.
        resi_connection:
            The convolutional block before residual connection.
            '1conv'/'3conv'
    """

    def __init__(
        self,
        params,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super(EncDec, self).__init__()
        self.img_range = params.img_range
        if params.in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = params.upscale
        self.window_size = params.window_size

        # 1, encoder
        self.conv_first = nn.Conv2d(
            params.in_chans, params.embed_dim, 3, 1, 1)

        # 2, decoder
        self.num_layers = len(params.depths)
        self.embed_dim = params.embed_dim
        self.ape = params.ape
        self.patch_norm = params.patch_norm
        self.num_features = params.embed_dim
        self.mlp_ratio = params.mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=(params.img_size_x, params.img_size_y),
            # Possible divisibility workaround (NOT implemented yet):
            # patch_size=1,
            patch_size=params.patch_size,
            in_chans=params.embed_dim,
            embed_dim=params.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=(params.img_size_x, params.img_size_y),
            # Possible divisibility workaround (NOT implemented yet):
            # patch_size=1,
            patch_size=params.patch_size,
            in_chans=params.embed_dim,
            embed_dim=params.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, params.embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=params.drop_rate)

        # stochastic depth
        dpr = [
            x.item()
            for x in torch.linspace(
                0, params.drop_path_rate, sum(params.depths))
        ]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dp_1 = sum(params.depths[:i_layer])
            dp_2 = sum(params.depths[: i_layer + 1])
            layer = RSTB(
                dim=params.embed_dim,
                input_resolution=(
                    patches_resolution[0], patches_resolution[1]),
                depth=params.depths[i_layer],
                num_heads=params.num_heads[i_layer],
                window_size=params.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=params.qkv_bias,
                qk_scale=params.qk_scale,
                drop=params.drop_rate,
                attn_drop=params.attn_drop_rate,
                drop_path=dpr[dp_1:dp_2],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=params.use_checkpoint,
                img_size=(params.img_size_x, params.img_size_y),
                # Possible divisibility workaround (NOT implemented yet):
                # patch_size=1,
                patch_size=params.patch_size,
                resi_connection=params.resi_connection,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if params.resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(
                params.embed_dim, params.embed_dim, 3, 1, 1
            )
        elif params.resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(
                    params.embed_dim, params.embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(
                    negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    params.embed_dim // 4, params.embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(
                    negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    params.embed_dim // 4, params.embed_dim, 3, 1, 1),
            )

        # 3, reconstruction
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(params.embed_dim, params.num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.upsample = Upsample(
            params.upscale, params.num_feat)
        self.conv_last = nn.Conv2d(
            params.num_feat, params.out_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (
            self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (
            self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class LayerScale2d(nn.Module):
    """Per-channel learnable gain on (B,C,H,W), init small.

    The body starts as a near-no-op and has to earn its contribution. Without it the
    optimizer's cheapest loss reduction, when the body is still noisy, is to zero out
    conv_after_body and let the skip carry the output -- which is what a model whose
    held-out temperature error equals HRRR's looks like (HANDOFF_multiscale_debug.md S5).
    """

    def __init__(self, dim, init=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1)


def _swin_stage(stage, f):
    """RSTB speaks tokens (B, H*W, C); these models speak images (B, C, H, W)."""
    b, c, h, w = f.shape
    t = f.flatten(2).transpose(1, 2)
    t = stage(t, (h, w))
    return t.transpose(1, 2).view(b, c, h, w)


class LayerNorm2d(nn.Module):
    """Per-pixel LayerNorm across channels only, on (B,C,H,W). ConvNeXt-style.

    Deliberately NOT GroupNorm, which normalizes over (C/G, H, W) and so (a) couples every
    pixel to every other through the shared spatial mean/variance -- measured: it saturates
    the receptive-field probe at any grid size, making the reach unmeasurable -- and (b)
    subtracts each sample's spatial field mean, in a network whose loss is 99.3% absolute
    field reconstruction. That second point is why EDSR removed BatchNorm from SR nets and
    why SwinIR, the body this repo is built on, has no norm in its conv path at all.

    Normalizing per pixel still does the one job the norm is here for: stop gain compounding
    through the resampling convs, which is what the pyramid's nine bare linear convs did
    (HANDOFF_multiscale_debug.md). Under channels_last the permutes are free.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _norm_act(dim, kind="layer", groups=8):
    if kind == "layer":
        norm = LayerNorm2d(dim)
    elif kind == "group":
        norm = nn.GroupNorm(math.gcd(groups, dim), dim)
    elif kind == "none":
        norm = nn.Identity()
    else:
        raise ValueError(f"stem_norm must be layer | group | none (got {kind!r})")
    return nn.Sequential(
        norm, nn.LeakyReLU(negative_slope=0.2, inplace=True))


class LowResEncDec(nn.Module):
    r"""Swin body at 1/``downscale`` resolution, convolutions everywhere else.

    Same contract as :class:`EncDec` -- (B, in_chans, H, W) -> (B, out_chans, H, W),
    residual vs HRRR -- but attention runs *only* on the downsampled grid. A window of
    ``w`` there spans ``w * downscale`` grid cells, so reach is bought where tokens are
    ``downscale**2`` times fewer, instead of at full res where the model is activation-
    memory-bound.

    Four deliberate design choices, each fixing a diagnosed failure of the earlier
    U-Net-pyramid attempt (the removed ``MultiScaleEncDec``; HANDOFF_lowres_arch.md S3):

    * every down/up stage is conv -> GroupNorm -> LeakyReLU, not a bare linear conv, so
      gain cannot compound through the resampling path;
    * the full-res skip is fused with a **3x3** conv, not a 1x1 -- a 1x1 has zero spatial
      extent and physically cannot spread an observation to its neighbours;
    * ``conv_after_body`` is gated by :class:`LayerScale2d`, and its residual is taken
      *inside* the body (``+ fc`` at 1/4 res); the full-res features arrive by concat, so
      there is no unmediated additive bypass around the whole network;
    * no attention at full resolution at all -- that was the ~11 GiB/sample line item that
      forced the pyramid's full-res decoder to depth 0.
    """

    def __init__(self, params, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.img_range = params.img_range
        self.upscale = params.upscale
        if self.upscale != 1:
            raise ValueError(
                "lowres: upscale must be 1 (the network has no super-resolution "
                f"stage; got {self.upscale})"
            )
        self.mean = torch.zeros(1, 1, 1, 1)

        self.downscale = int(getattr(params, "downscale", 4))
        if self.downscale not in (2, 4, 8):
            raise ValueError(
                f"lowres: downscale must be 2, 4 or 8 (got {self.downscale})")
        n_down = int(math.log2(self.downscale))

        stem_dims = list(getattr(params, "stem_dims", [64, 96]))
        if len(stem_dims) != n_down:
            raise ValueError(
                f"lowres: stem_dims needs {n_down} entries for downscale "
                f"{self.downscale} (conv_first output, then one per 2x stage below the "
                f"top); got {len(stem_dims)}"
            )
        body_dim = int(getattr(params, "body_dim", 128))
        depths = list(params.depths)
        heads = list(params.num_heads)
        if len(heads) != len(depths):
            raise ValueError(
                f"lowres: num_heads must have one entry per RSTB group "
                f"({len(depths)}); got {len(heads)}"
            )
        for h in heads:
            if body_dim % h != 0:
                raise ValueError(
                    f"lowres: body_dim {body_dim} not divisible by {h} heads")

        ws = params.window_size
        if isinstance(ws, (list, tuple)):
            raise ValueError(
                "lowres: window_size is a scalar here (the pyramid's per-level list has "
                f"no meaning with a single body resolution); got {ws}"
            )
        self.window_size = int(ws)

        # The body sees H/downscale, and its windows must tile it -> H must be a multiple
        # of downscale * window_size. 1356 and 2294 are multiples of neither, so
        # check_image_size below always fires.
        self.pad_multiple = self.downscale * self.window_size
        pad_h = self._round_up(params.img_size_y, self.pad_multiple)
        pad_w = self._round_up(params.img_size_x, self.pad_multiple)
        # (H, W) order, matching the x_size forward() threads through -- this is what makes
        # SwinTransformerBlock hit its precomputed-attn_mask fast path.
        self.body_resolution = (pad_h // self.downscale, pad_w // self.downscale)

        # ---- stem: full res -> 1/downscale ----
        self.conv_first = nn.Conv2d(params.in_chans, stem_dims[0], 3, 1, 1)

        nk = str(getattr(params, "stem_norm", "layer")).lower()
        down = []
        for d_in, d_out in zip(stem_dims, stem_dims[1:] + [body_dim]):
            down += [nn.Conv2d(d_in, d_out, 3, 2, 1), _norm_act(d_out, nk)]
        self.down = nn.Sequential(*down)

        # ---- body: the only attention in the network ----
        total_depth = sum(depths)
        dpr = [
            x.item()
            for x in torch.linspace(0, params.drop_path_rate, total_depth)
        ]
        self.body = nn.ModuleList()
        cursor = 0
        for i, d in enumerate(depths):
            self.body.append(
                RSTB(
                    dim=body_dim,
                    input_resolution=self.body_resolution,
                    depth=d,
                    num_heads=heads[i],
                    window_size=self.window_size,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    qk_scale=params.qk_scale,
                    drop=params.drop_rate,
                    attn_drop=params.attn_drop_rate,
                    drop_path=dpr[cursor: cursor + d],
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=params.use_checkpoint,
                    img_size=self.body_resolution,
                    patch_size=1,
                    resi_connection=params.resi_connection,
                )
            )
            cursor += d

        self.conv_after_body = nn.Conv2d(body_dim, body_dim, 3, 1, 1)
        self.layer_scale = LayerScale2d(
            body_dim, float(getattr(params, "layer_scale_init", 1e-4)))

        # ---- up: 1/downscale -> full res, PixelShuffle by 2 each stage ----
        num_feat = params.num_feat
        up = []
        for d_in in [body_dim] + [num_feat] * (n_down - 1):
            up += [nn.Conv2d(d_in, num_feat * 4, 3, 1, 1),
                   nn.PixelShuffle(2),
                   _norm_act(num_feat, nk)]
        self.up = nn.Sequential(*up)

        # ---- full-res head: 3x3 convs, NOT the pyramid's 1x1 fuse ----
        head_dims = list(getattr(params, "head_dims", [64, 64]))
        head = []
        for d_in, d_out in zip([num_feat + stem_dims[0]] + head_dims[:-1], head_dims):
            head += [nn.Conv2d(d_in, d_out, 3, 1, 1),
                     nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.head = nn.Sequential(*head)
        self.conv_last = nn.Conv2d(head_dims[-1], params.out_chans, 3, 1, 1)

        self.apply(self._init_weights)
        # After apply(): _init_weights does not touch nn.Parameter directly, but a future
        # edit to it might, and the small init is the whole point of the gate.
        nn.init.constant_(
            self.layer_scale.gamma, float(getattr(params, "layer_scale_init", 1e-4)))

    @staticmethod
    def _round_up(v, m):
        return ((v + m - 1) // m) * m

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        m = self.pad_multiple
        return F.pad(
            x, (0, (m - w % m) % m, 0, (m - h % m) % m), "reflect")

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        f0 = self.conv_first(x)          # full res, kept for the head's concat
        fc = self.down(f0)               # 1/downscale

        f = fc
        for stage in self.body:
            f = _swin_stage(stage, f)
        f = self.layer_scale(self.conv_after_body(f)) + fc

        f = self.up(f)                   # back to full res
        x = self.conv_last(self.head(torch.cat([f, f0], dim=1)))

        x = x / self.img_range + self.mean

        return x[:, :, :H, :W]


def build_model(params):
    """Single entry point: flat SwinIR | 1/4-res Swin body."""
    arch = str(getattr(params, "arch", "") or "").lower()
    if not arch:  # back-compat: configs written before `arch` existed
        arch = "pyramid" if bool(getattr(params, "multiscale", False)) else "flat"
    if arch == "lowres":
        return LowResEncDec(params)
    if arch == "pyramid":
        raise ValueError(
            "arch 'pyramid' (MultiScaleEncDec) was removed -- it was the diagnosed "
            "failure that 'lowres' (LowResEncDec) replaced. Use arch: lowres."
        )
    if arch != "flat":
        raise ValueError(f"unknown arch {arch!r} (flat | lowres)")
    return EncDec(params)


if __name__ == "__main__":

    params = {
        "upscale": 1,
        "in_chans": 8,
        "out_chans": 4,
        "img_size_x": 960,
        "img_size_y": 480,
        "window_size": 4,
        "patch_size": 5,
        "num_feat": 64,
        "drop_rate": 0.1,
        "drop_path_rate": 0.1,
        "attn_drop_rate": 0.1,
        "ape": False,
        "patch_norm": True,
        "use_checkpoint": False,
        "resi_connection": "1conv",
        "qkv_bias": True,
        "qk_scale": None,
        "img_range": 1.0,
        "depths": [3],  # [3]
        "embed_dim": 64,  # need be divisible by num_heads
        "num_heads": [4],
        "mlp_ratio": 2,
    }
    import argparse

    params = argparse.Namespace(**params)
    model = EncDec(
        params,
    )

    print(model)

    x = torch.randn((1, 8, 960, 480))
    x = model(x)
    print(x.shape)
