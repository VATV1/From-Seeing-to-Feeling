import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from models.clip import clip
from models.cocoop import TextEncoder, PromptLearner
from models.seg_decoder import SegDecoder
from torchvision.ops import DeformConv2d

from models.SEM import Semantic_Enhancement_Module


from PIL import Image
import math


import torch

def integrate_masks(pred):
    batch_size, num_masks, size, _ = pred.shape
    integrated_masks = torch.zeros((batch_size, 1, size, size), dtype=pred.dtype, device=pred.device)
    for b in range(batch_size):
        for i in range(size):
            for j in range(size):
                pixel_values = pred[b, :, i, j]
                if torch.any(pixel_values > 0):
                    integrated_masks[b, 0, i, j] = torch.max(pixel_values)
    return integrated_masks


def count_non_zero_pixels(target):
    target=target.cpu()
    image_array = np.array(target)
    print(np.sum(image_array > 0))
    return np.sum(image_array > 0)

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiScaleHeatmapGenerator(nn.Module):
    def __init__(self, num_keypoints=15, scales=[1.0, 2.0, 4.0]):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.scales = scales

        # 为每个尺度创建高斯核
        self.kernels = []
        for scale in scales:
            kernel_size = int(6 * scale) + 1
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            kernel = self.create_gaussian_kernel(kernel_size, scale)
            self.kernels.append(kernel)

        # 尺度权重学习
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))

    def create_gaussian_kernel(self, size, sigma):
        ax = torch.linspace(-(size // 2), size // 2, size)
        xx, yy = torch.meshgrid(ax, ax, indexing='xy')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        return kernel / kernel.max()  # 归一化

    def forward(self, image_tensor, keypoints):
        B, C, H, W = image_tensor.shape

        # 初始化热图
        heatmaps = torch.zeros(B, self.num_keypoints, H, W, device=image_tensor.device)

        for b in range(B):
            for kp_idx in range(self.num_keypoints):
                x, y = keypoints[b, kp_idx]
                x, y = int(x), int(y)

                # 应用多尺度核
                for i, (scale, kernel) in enumerate(zip(self.scales, self.kernels)):
                    pad = kernel.size(0) // 2

                    if x - pad < 0 or y - pad < 0 or x + pad >= W or y + pad >= H:
                        continue

                    patch = heatmaps[b, kp_idx, y - pad:y + pad + 1, x - pad:x + pad + 1]
                    weighted_kernel = kernel.to(image_tensor.device) * self.scale_weights[i]
                    heatmaps[b, kp_idx, y - pad:y + pad + 1, x - pad:x + pad + 1] = torch.maximum(
                        patch, weighted_kernel
                    )

        return heatmaps
class Net(nn.Module):
    def __init__(self, args, input_dim, out_dim, dino_pretrained='dinov2_vitb14'):
        super().__init__()
        self.dino_pretrained = dino_pretrained
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.class_names = args.class_names
        self.num_aff = len(self.class_names)

        # set up a vision embedder
        self.embedder = Mlp(in_features=input_dim, hidden_features=int(out_dim), out_features=out_dim,
                            act_layer=nn.GELU, drop=0.)
        self.dino_model = torch.hub.load('facebookresearch/dinov2', self.dino_pretrained).cuda()

        clip_model = load_clip_to_cpu('ViT-B/16').float()
        classnames = [a.replace('_', ' ')for a in self.class_names]
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts              
        self.aff_text_encoder = TextEncoder(clip_model)

        self.seg_decoder = SegDecoder(embed_dims=out_dim, num_layers=2)

        self.merge_weight = nn.Parameter(torch.zeros(3))

        self.lln_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)])
        self.lln_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(3)])

        self.lln_norm_1 = nn.LayerNorm(out_dim)
        self.lln_norm_2 = nn.LayerNorm(out_dim)

        self.linear_cls = nn.Linear(input_dim, out_dim)

        # ------------------ 初始化SEM ----------------------------
        self.sem = Semantic_Enhancement_Module(num_class=self.num_aff)
        # --------------------------------------------------------

        self._freeze_stages(exclude_key=['embedder', 'ctx', 'seg_decoder', 'lln_', 'merge_weight', 'linear_cls'])
        # self.key=MultiScaleHeatmapGenerator()
        self.step=0

    def anisotropic_oriented_mask_fast(self,
            image_tensor,  # (B, C, H, W)
            keypoints,  # either tensor (B, N, 2) or list of tensors [(N_i,2), ...]
            patch_k=15,  # kernel size for oriented gaussian (odd)
            struct_s=15,  # patch size for structure tensor smoothing (odd)
            base_sigma=1.2,
            sigma_gain=1.8,
            normalize=True,
            chunk_size=1024  # if N*k*k too big, process kps in chunks to reduce memory
    ):
        """
        Vectorized, faster anisotropic oriented gaussian renderer.

        Returns: masks (B,1,H,W) dtype=float32 on same device as image_tensor.
        """
        device = image_tensor.device
        B, C, H, W = image_tensor.shape
        k = patch_k
        half_k = k // 2
        half_s = struct_s // 2
        eps = 1e-9

        # Make keypoints into list per batch for flexible inputs
        if isinstance(keypoints, torch.Tensor):
            # assume shape (B, N, 2)
            kps_list = [keypoints[b] for b in range(keypoints.shape[0])]
        else:
            kps_list = list(keypoints)

        # 1) compute grayscale & global gradients once per batch
        if C == 1:
            gray = image_tensor
        else:
            gray = image_tensor.mean(dim=1, keepdim=True)  # (B,1,H,W)

        # sobel filters (device)
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]], device=device).view(1, 1, 3, 3)

        Ix = F.conv2d(gray, sobel_x, padding=1)  # (B,1,H,W)
        Iy = F.conv2d(gray, sobel_y, padding=1)  # (B,1,H,W)
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy

        # smooth structure tensor components with a box filter (fast)
        box = torch.ones(1, 1, struct_s, struct_s, device=device) / (struct_s * struct_s)
        pad = struct_s // 2
        Sxx = F.conv2d(Ix2, box, padding=pad)  # (B,1,H,W)
        Syy = F.conv2d(Iy2, box, padding=pad)
        Sxy = F.conv2d(Ixy, box, padding=pad)

        # eigen-quantities per-pixel (lam1, lam2, theta)
        # compute as tensors (B,1,H,W)
        diff = Sxx - Syy
        # disc = sqrt((diff/2)^2 + Sxy^2)
        disc = torch.sqrt((diff * 0.5) ** 2 + Sxy * Sxy + eps)
        tr = (Sxx + Syy) * 0.5
        lam1_map = tr + disc  # (B,1,H,W)
        lam2_map = tr - disc
        theta_map = 0.5 * torch.atan2(2.0 * Sxy, diff + eps)  # principal direction

        # precompute kernel grid coords centered
        ax = torch.arange(-half_k, half_k + 1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ax, ax, indexing='ij')  # (k,k) floats
        # reshape small arrays for broadcasting later
        xx = xx.view(1, k, k)  # (1,k,k)
        yy = yy.view(1, k, k)

        out_masks = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)

        # For each batch element: vectorized over its N keypoints (in chunks if necessary)
        for b in range(B):
            kps = kps_list[b]
            if kps is None or kps.numel() == 0:
                continue

            # ensure float coordinates (N,2)
            kps_xy = kps[:, :2].float().to(device)  # x,y
            N = kps_xy.shape[0]

            # sample lam1/lam2/theta at keypoint positions via grid_sample (vectorized)
            # build normalized grid coords in [-1,1]
            x = kps_xy[:, 0]
            y = kps_xy[:, 1]
            x_norm = (x / max(W - 1, 1)) * 2.0 - 1.0
            y_norm = (y / max(H - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([x_norm, y_norm], dim=1).view(1, N, 1, 2)  # (1,N,1,2)

            # sample maps (each of shape (1,1,H,W)), get (1,1,N,1) -> squeeze to (N,)
            lam1_k = F.grid_sample(lam1_map[b:b + 1], grid, align_corners=True, mode='bilinear').view(-1)
            lam2_k = F.grid_sample(lam2_map[b:b + 1], grid, align_corners=True, mode='bilinear').view(-1)
            theta_k = F.grid_sample(theta_map[b:b + 1], grid, align_corners=True, mode='bilinear').view(-1)

            # compute ratio & anisotropic sigmas for each keypoint
            lam1_k = lam1_k.clamp(min=0.0)
            lam2_k = lam2_k.clamp(min=0.0)
            ratio = (lam1_k / (lam2_k + eps)).clamp(min=1.0)
            # map ratio to sigma_major (vectorized)
            sigma_major = base_sigma * (1.0 + sigma_gain * (ratio - 1.0) / (ratio + 1.0))
            sigma_minor = torch.full_like(sigma_major, fill_value=float(base_sigma))

            # process in chunks if N*k*k too large
            cells_per_kp = k * k
            max_cells = chunk_size
            step = max(1, max_cells // cells_per_kp)  # how many keypoints per chunk
            # step at least 1
            for s_idx in range(0, N, step):
                e_idx = min(N, s_idx + step)
                cur_kps = kps_xy[s_idx:e_idx]  # (M,2)
                M = cur_kps.shape[0]
                # gather per-kp params
                sigma_M = sigma_major[s_idx:e_idx].view(M, 1, 1)  # (M,1,1)
                sigma_m = sigma_minor[s_idx:e_idx].view(M, 1, 1)
                theta_M = theta_k[s_idx:e_idx].view(M, 1, 1)
                # cos/sin (M,1,1)
                cos_t = torch.cos(theta_M)
                sin_t = torch.sin(theta_M)
                # broadcast rotate coords: result (M,k,k)
                x_rot = cos_t * xx - sin_t * yy  # (M,k,k)
                y_rot = sin_t * xx + cos_t * yy
                # compute anisotropic gaussian (M,k,k)
                denom_x = 2.0 * (sigma_M * sigma_M) + eps
                denom_y = 2.0 * (sigma_m * sigma_m) + eps
                g = torch.exp(-(x_rot * x_rot) / denom_x - (y_rot * y_rot) / denom_y)  # (M,k,k)
                # normalize each kernel so peak ~1
                g = g / (g.amax(dim=[1, 2], keepdim=True) + eps)

                # compute absolute positions for every kernel cell: (M,k,k)
                # offsets for kernel cells: rx, ry = [-half_k .. +half_k]
                rx = (xx.view(1, k, k))  # (1,k,k) same as xx
                ry = (yy.view(1, k, k))
                # cell absolute coords = kp + (rx, ry)
                kp_x = cur_kps[:, 0].view(M, 1, 1)
                kp_y = cur_kps[:, 1].view(M, 1, 1)
                cell_x = kp_x + rx  # (M,k,k)
                cell_y = kp_y + ry

                # floor indices
                ix = torch.floor(cell_x).long()  # (M,k,k)
                iy = torch.floor(cell_y).long()
                # fractional
                dx = (cell_x - ix.float()).clamp(0.0, 1.0)
                dy = (cell_y - iy.float()).clamp(0.0, 1.0)
                w00 = (1.0 - dx) * (1.0 - dy)
                w10 = dx * (1.0 - dy)
                w01 = (1.0 - dx) * dy
                w11 = dx * dy

                # four neighbor indices (clamp to image)
                ix00 = ix.clamp(0, W - 1)
                iy00 = iy.clamp(0, H - 1)
                ix10 = (ix + 1).clamp(0, W - 1)
                iy10 = iy00
                ix01 = ix00
                iy01 = (iy + 1).clamp(0, H - 1)
                ix11 = ix10
                iy11 = iy01

                # flatten arrays and compute linear indices: idx = iy * W + ix
                def flat_idx(ix_t, iy_t):
                    return (iy_t.view(-1).to(torch.long) * W + ix_t.view(-1).to(torch.long))

                idx00 = flat_idx(ix00, iy00)
                idx10 = flat_idx(ix10, iy10)
                idx01 = flat_idx(ix01, iy01)
                idx11 = flat_idx(ix11, iy11)

                # contributions (flatten)
                base = g.view(-1)  # (M*k*k,)
                contrib00 = (base * w00.view(-1))
                contrib10 = (base * w10.view(-1))
                contrib01 = (base * w01.view(-1))
                contrib11 = (base * w11.view(-1))

                # combine indices and contributions
                all_idx = torch.cat([idx00, idx10, idx01, idx11], dim=0)
                all_contrib = torch.cat([contrib00, contrib10, contrib01, contrib11], dim=0)

                # accumulate into linear image vector (H*W)
                out_flat = torch.zeros(H * W, device=device, dtype=torch.float32)
                # index_add_ will sum contributions to same index
                out_flat.index_add_(0, all_idx, all_contrib)

                # reshape to H,W and add to out_masks[b]
                out_patch = out_flat.view(H, W)
                # accumulate into output (in case multiple chunks)
                out_masks[b, 0] += out_patch

            # normalization per image if desired
            if normalize:
                s = out_masks[b, 0].sum()
                if s > 0:
                    out_masks[b, 0] /= (s + eps)

        return out_masks  # (B,1,H,W)


    def get_orb_mask(self, image_tensor, keypoints):
        original_device = image_tensor.device
        b, _, h, w = image_tensor.shape
        masks = []
        gaussian = self.gaussian_kernel().to(original_device)
        pad = 4  # 9//2

        for i in range(b):
            mask = torch.zeros(h, w, device=original_device)
            kp_coords = keypoints[i]  # (N, 2)

            for x, y in kp_coords:
                x, y = int(x), int(y)
                if x-pad < 0 or y-pad < 0 or x+pad >= w or y+pad >= h:
                    continue
                current_patch = mask[y-pad:y+pad+1, x-pad:x+pad+1]
                mask[y-pad:y+pad+1, x-pad:x+pad+1] = torch.maximum(current_patch, gaussian).to(original_device)

            masks.append(torch.clip(mask, 0, 2))

        output_tensor = torch.stack(masks).unsqueeze(1).to(original_device)
        return output_tensor

    def gaussian_kernel(self, size=9, sigma=1.0):
        ax = torch.linspace(-(size // 2), size // 2, size)
        # xx, yy = torch.meshgrid(ax, ax)
        xx, yy = torch.meshgrid(ax, ax, indexing='xy')
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel
    
    def save_pic(self,target):
        target[target>0]=255
        image=Image.fromarray(target.cpu().numpy(),mode='L')
        path="./pic/"+str(self.step)+".png"
        self.step+=1
        image.save(path)
        print("图片已经保存")
        print("图片路径：{}".format(path))


    def forward(self, img, keypoints=None, label=None, gt_aff=None):
        b, _, h, w = img.shape

        # DINO特征提取
        dino_out = self.dino_model.get_intermediate_layers(img, n=3, return_class_token=True)
        merge_weight = torch.softmax(self.merge_weight, dim=0)
        
        dino_dense = 0
        for i, feat in enumerate(dino_out):
            feat_ = self.lln_linear[i](feat[0])
            feat_ = self.lln_norm[i](feat_)
            dino_dense += feat_ * merge_weight[i]
        dino_dense = self.lln_norm_1(self.embedder(dino_dense))

        # -----------------sem---------------------
        dino_dense = self.sem(dino_dense, label)

        # -----------------------------------------

        # prompts = self.prompt_learner()
        # ----------------cocoop-------------------------
        prompts0 = self.prompt_learner(dino_dense)
        prompts = prompts0.squeeze(0)
        # -----------------------------------------------
        tokenized_prompts = self.tokenized_prompts
        text_features = self.lln_norm_2(self.aff_text_encoder(prompts, tokenized_prompts))

        dino_cls = dino_out[-1][1]
        dino_cls = self.linear_cls(dino_cls)

        text_features = text_features.unsqueeze(0).expand(b, -1, -1)  
        text_features, attn_out, _ = self.seg_decoder(text_features, dino_dense, dino_cls)

        
        attn = (text_features[-1] @ dino_dense.transpose(-2,-1)) * (512**-0.5)
        # print(text_features[-1].shape)
        # print(dino_dense.transpose(-2,-1).shape)

        attn_out = torch.sigmoid(attn)
        attn_out = attn_out.reshape(b, -1, h // 14, w // 14)
        pred = F.interpolate(attn_out, img.shape[-2:], mode='bilinear', align_corners=False)


        if keypoints is not None:
            orb_mask = self.anisotropic_oriented_mask_fast(img, keypoints)
            # orb_mask = self.get_orb_mask(img, keypoints)

            orb_mask = F.interpolate(orb_mask, img.shape[-2:], mode='bilinear', align_corners=False)

            scaled_orb_mask = 1 + (orb_mask - orb_mask.min()) / (orb_mask.max() - orb_mask.min() + 1e-8)

            pred = scaled_orb_mask * pred
            pred = torch.clamp(pred, 0, 1)

        
        if self.training:
            assert not label == None, 'Label should be provided during training'
            loss_bce = nn.BCELoss()(pred, label / 255.0)
            loss_dict = {'bce': loss_bce}
            return pred, loss_dict

        else:
            if gt_aff is not None:
                out = torch.zeros(b, h, w).cuda()
                for b_ in range(b):
                    out[b_] = pred[b_, gt_aff[b_]]
                return out

    def _freeze_stages(self, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in self.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False