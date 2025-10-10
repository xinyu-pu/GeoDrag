from math import tau
import cv2
import torch
import numpy as np
import copy

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
# = = = = = = = Displacement Field Computation = = = = = = = = #
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

pdist = torch.nn.PairwiseDistance(p=2)

def get_rectangle(mask: torch.Tensor):
    """Compute the bounding rectangle of the nonzero (True) region in a binary mask.
    
    Args:
        mask (Bool[Tensor, 'H W']): 
            A binary mask tensor where nonzero (True) elements represent the region of interest. 
            The tensor is expected to be 2D, with shape (height, width).

    Returns:
        rect (Float[Tensor, '4 2']): 
            A tensor containing the coordinates of the four corners of the bounding rectangle.
            The corners are ordered as [top-left, top-right, bottom-left, bottom-right].
        left_top (Float[Tensor, '2']): Coordinates of the top-left corner (y, x).
        left_bottom (Float[Tensor, '2']): Coordinates of the bottom-left corner (y, x).
        right_top (Float[Tensor, '2']): Coordinates of the top-right corner (y, x).
        right_bottom (Float[Tensor, '2']): Coordinates of the bottom-right corner (y, x).
    """
    index_1 = torch.nonzero(mask)   
    min_y,min_x = torch.min(index_1,dim=0)[0][-2:]
    max_y,max_x = torch.max(index_1,dim=0)[0][-2:]
    left_top = torch.Tensor((min_y, min_x)).to(device=mask.device)
    left_bottom = torch.Tensor((min_y, max_x)).to(device=mask.device)
    right_top = torch.Tensor((max_y, min_x)).to(device=mask.device)
    right_bottom = torch.Tensor((max_y, max_x)).to(device=mask.device)
    rect = torch.stack((left_top, left_bottom, right_top, right_bottom),dim=0).to(device=mask.device)
    return rect, left_top, left_bottom, right_top, right_bottom

def get_circle(mask: torch.Tensor):
    """Compute the center and radius of the minimum enclosing circle for a binary mask.
    
    Args:
        mask (Bool[Tensor, '1 1 H W']): 
            A binary mask tensor where nonzero (True) elements represent the region of interest. 
            The tensor is expected to be 4D with shape (1, 1, height, width).
            
    Returns:
        center (Float[Tensor, '2']): Coordinates of the circle center (y, x).
        radius (float): Radius of the circle.
    
    """
    rect, left_top, left_bottom, right_top, right_bottom = get_rectangle(mask=mask)
    center = torch.Tensor(((left_top[0] + right_bottom[0]) / 2, (left_top[1] + right_bottom[1]) / 2)).to(device=mask.device)  # y,x
    radius = pdist(center, left_top) 
    return center,radius

"""
NOTE: The following two functions (estimate_beta_from_depth and estimate_alpha_from_depth) are
empirical designs to adaptively adjust the parameters based on depth statistics. However, they are not
strictly necessary for the method to work, and we do not use them in all experiments.
"""
def estimate_alpha_from_depth(depth, mask, beta=4, center_contrast=1.0, min_alpha=0.5, max_alpha=1.2):
    masked_depth = depth[mask.bool()]
    if masked_depth.numel() == 0:
        return 1.0  # fallback
    depth_std = torch.std(masked_depth)
    depth_mean = torch.mean(masked_depth)
    contrast = depth_std / (depth_mean + 1e-6)  # depth variation ratio
    # Normalize contrast to [0, 1]
    norm_c = torch.sigmoid(beta * (contrast - center_contrast))  # adjust 4.0 or center as needed
    alpha = min_alpha + (max_alpha - min_alpha) * norm_c

    return alpha.item()

def estimate_beta_from_depth(depth: torch.Tensor,
                              mask: torch.Tensor,
                              center_contrast=0.08,
                              slope=3.5,
                              min_beta=1.0,
                              max_beta=1.7) -> float:
    masked_depth = depth[mask.bool()]
    if masked_depth.numel() == 0:
        return 1.8  # fallback

    std = torch.std(masked_depth)
    mean = torch.mean(masked_depth)
    contrast = std / (mean + 1e-6)
    norm_c = torch.sigmoid(-slope * (contrast - center_contrast))
    beta = min_beta + (max_beta - min_beta) * norm_c
    return beta.item()

# Influence functions
def plane_aware_influence_fun(
    handle, 
    mask, 
    H, W, O, R, 
    beta, 
    device, 
    mode='linear', # FIXME: 'linear'
    depth=None
):
    O = O.flip(0)
    index_1 = torch.nonzero(mask, as_tuple=False)  # [N, 2]
    grid = index_1[:, [1, 0]].float()  

    # Calculate the maximum distance from the handle to the mask boundary
    delta = grid - handle[None, None, :] 
    delta_norm = delta.norm(dim=-1) + 1e-6
    OA = O - handle # [2]
    OA_norm = OA.norm(dim=-1) + 1e-6
    cos_theta = (delta @ OA) / (delta_norm * OA_norm)  
    sign_map = torch.where(cos_theta > 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))  # [H, W]
    L = R + sign_map * OA_norm # [H, W]
    L = torch.clamp(L, min=R)  
    
    if mode == 'auto': # NOTE: Do not use 'auto' mode in all experiments, as it may lead to unstable results.
        beta = estimate_beta_from_depth(depth, mask, center_contrast=0.5, slope=beta, min_beta=1.15, max_beta=1.8)
        print(f"Automatically estimated beta: {beta}")

    # Compute the influence weight
    weight = 1 - (delta_norm / L) ** beta
    
    weight_full = torch.zeros((H, W), device=device)
    weight_full[index_1[:, 0], index_1[:, 1]] = weight
    return weight_full

# Geometry-aware influence function
def geometry_aware_influence_fun(
    depth, 
    mask, 
    handle, 
    upper_scale=5.0, 
    lower_scale=0.1, 
    mode='linear',  # FIXME: 'linear'
    alpha=1.0
):
    x, y = handle.long()
    d_h = depth[y, x].clamp(min=1e-4)
    if mode == 'auto': # NOTE: Do not use 'auto' mode in all experiments, as it may lead to unstable results.
        alpha = estimate_alpha_from_depth(depth, mask, beta=alpha)
        print(f"Automatically estimated alpha: {alpha}")
    f = (depth / d_h) ** alpha
    f = torch.clamp(f, min=lower_scale, max=upper_scale)
    f[mask == 0] = 0.0
    return f

# Displacement field computation
def displacement_field_computation(
    depth_predictor,
    source_image, 
    mask, 
    handle_points, 
    target_points,
    gamma_ratio=0.5,
    upper_scale=5,
    lower_scale=1e-5,
    alpha=1.0,
    beta=1.0,
    scale=1,
    **kwargs
):   
    """Compute the displacement field based on user-defined point drags.
    
    Args:
        depth_predictor: The depth prediction model. In our experiments, we use Depth Anything V2.
        source_image (Float[np.ndarray, 'H W C']): The source image in numpy array format.
        mask (Bool[torch.Tensor, '1 1 H W']): The mask indicating the region of interest.
        handle_points (Float[torch.Tensor, 'N 2']): List of handle points.
        target_points (Float[torch.Tensor, 'N 2']): List of target points.
        gamma_ratio (float): The gamma ratio for displacement field computation. Defaults to 0.5.
        upper_scale (float): The upper scale for geometry-aware influence function. Defaults to 5.
        lower_scale (float): The lower scale for geometry-aware influence function. Defaults to 1e-5.
        alpha (float): The alpha parameter for geometry-aware influence function. Defaults to 1.0.
        beta (float): The beta parameter for plane-aware influence function. Defaults to 1.0.
        scale (float): The overall scale for the displacement field. Defaults to 1.0.
        **kwargs: Additional keyword arguments.
    
    Returns:
        Float[torch.Tensor, 'H W 2']: The computed displacement field.
    """
    device = mask.device
    
    # depth prediction
    raw_img = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    depth_map = depth_predictor.infer_image(raw_img) # HxW raw depth map in numpy
    latent_H, latent_W = mask.shape[-2:]
    depth_map  = cv2.resize(depth_map , (latent_W, latent_H), interpolation=cv2.INTER_LINEAR)
    depth_map = torch.from_numpy(depth_map).to(device) # HxW
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # conflict-free partition
    N = handle_points.shape[0]
    flow = torch.zeros(latent_H, latent_W, 2, device=device)
    yy, xx = torch.meshgrid(torch.arange(latent_H, device=device), torch.arange(latent_W, device=device), indexing='ij')
    pixel_grid = torch.stack([xx, yy], dim=-1).float()  # [H, W, 2]
    mask = mask.squeeze(0).squeeze(0).bool()

    masked_coords = mask.nonzero(as_tuple=False)  # [M, 2] -> (y, x)
    coords_xy = masked_coords[:, [1, 0]].float().to(device)  # [M, 2] (x, y)
    distances = torch.norm(coords_xy[:, None, :] - handle_points[None, :, :], dim=-1)
    assignments = torch.argmin(distances, dim=-1) # [M]
    
    handle_points = handle_points.float()
    target_points = target_points.float()
    for i in range(N):
        sub_mask = torch.zeros(latent_H, latent_W, dtype=torch.bool, device=device)
        indices_i = masked_coords[assignments == i]
        sub_mask[indices_i[:, 0], indices_i[:, 1]] = True
        
        handle = handle_points[i] 
        target = target_points[i]
        O, R = get_circle(mask=sub_mask.unsqueeze(0).unsqueeze(0))
        gamma = gamma_ratio * 2 * R
        
        direction = target - handle
        if direction.norm() < 1e-4:
            continue
        pixel_distance_from_handle = torch.norm(pixel_grid - handle[None, None, :], dim=-1)
        
        DF_weight_plane = plane_aware_influence_fun(
            handle, 
            sub_mask, 
            latent_H, 
            latent_W, 
            O, 
            R, 
            beta, 
            device, 
            mode='linear')  # [N, H, W]
        
        DF_weight_depth = geometry_aware_influence_fun(
            depth_map, 
            sub_mask, 
            handle, 
            upper_scale, 
            lower_scale, 
            alpha=alpha,
            mode='linear')  # [N, H, W]
        
        lambda_fusion = pixel_distance_from_handle / (pixel_distance_from_handle + gamma + 1e-6)  # [N, H, W]
        DF_weight_fusion = (1 - lambda_fusion) * DF_weight_plane + lambda_fusion * DF_weight_depth  # [N, H, W]
        
        direction = direction.view(1, 1, 2) 
        
        move_vector = DF_weight_fusion.unsqueeze(-1) * direction * scale
        
        flow[sub_mask] = move_vector[sub_mask]
        
    return flow

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
# = = = = = = = = = = = = interpolation = = = = = = = = = = = = #
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
"""
There are two interpolation functions implemented here:

1. **BNNI interpolation (dynamic version)**  
   This is the original algorithm used in FastDrag.  
   It updates each position dynamically, meaning that once a value is
   interpolated and written back into the tensor, it can immediately
   serve as a valid neighbor for subsequent positions. This ensures
   that the interpolation respects sequential dependencies.

2. **Vectorized interpolation (static version)**  
   This re-implements the interpolation in a fully vectorized way.  
   Instead of updating values position by position, it computes the
   nearest neighbors and interpolation weights for the entire grid
   in a single pass, and then updates all missing positions at once.
   This breaks the sequential dependency: newly interpolated values
   are not reused within the same pass.  

NOTE: As a result, the two methods are *not numerically equivalent*.
However, the vectorized version is much faster in practice, at the
cost of deviating from the exact original FastDrag interpolation
semantics.
"""

def interpolation_dynamic(x, ori_x, appeared_mask=None):
    """Dynamic interpolation for missing values in the input tensor.

    Args:
        x (Float[Tensor, 'B C N M']): The input tensor with potential missing values.
        ori_x (Float[Tensor, 'B C N M']): The original tensor for reference.
        appeared_mask (Bool[Tensor, 'N M'], optional): A mask indicating which values have appeared. Defaults to None.

    Returns:
        Float[Tensor, 'B C N M']: The interpolated tensor.
        Bool[Tensor, 'N M']: A mask indicating which positions were interpolated.
    
    NOTE: This interpolation method dynamically updates the tensor,
    meaning that once a value is interpolated and written back into the tensor,
    it can immediately serve as a valid neighbor for subsequent positions. This ensures
    that the interpolation respects sequential dependencies. However, this also means that
    the order of processing can affect the results, and the time overhead is higher than static methods (interpolation_static).
    """

    assert x.dim() == 4, "Input tensor x should have shape (B, C, N, M)"
    assert x.shape == ori_x.shape
    batch_size, channels, N, M = x.shape 
    device = x.device
    appeared_mask = appeared_mask.squeeze().bool() if appeared_mask is not None else torch.zeros((N, M), dtype=torch.bool, device=device)
    mask = (x[:, 0] == 0)  # Shape: (batch_size, N, M)
    
    for b in range(batch_size):
        zero_positions = (x[b, 0] == 0)

        for i in range(N):
            for j in range(M):
                if zero_positions[i, j]:
                    values = []  
                    weights = [] 
                    
                    use_ori = appeared_mask[i, j]
                    ref_source = ori_x if use_ori else x
                    # Search in four directions
                    for k in range(1, j + 1): 
                        if j - k >= 0 and ref_source[b, 0, i, j - k] != 0:
                            values.append(ref_source[b, :, i, j - k])
                            weights.append(1 / k)
                            break

                    for k in range(1, M - j):
                        if j + k < M and ref_source[b, 0, i, j + k] != 0:
                            values.append(ref_source[b, :, i, j + k])
                            weights.append(1 / k)
                            break

                    for k in range(1, i + 1):
                        if i - k >= 0 and ref_source[b, 0, i - k, j] != 0:
                            values.append(ref_source[b, :, i - k, j])
                            weights.append(1 / k)
                            break

                    for k in range(1, N - i):
                        if i + k < N and ref_source[b, 0, i + k, j] != 0:
                            values.append(ref_source[b, :, i + k, j])
                            weights.append(1 / k)
                            break

                    if weights:
                        total_weight = sum(weights)
                        interpolated_value = sum(w * v for w, v in zip(weights, values)) / total_weight
                        x[b, :, i, j] = interpolated_value

    return x, mask

def interpolation_static(x, ori_x, appeared_mask=None):
    """Static interpolation method for filling missing values in the input tensor.
    
     Args:
        x (Float[Tensor, 'B C N M']): The input tensor with potential missing values.
        ori_x (Float[Tensor, 'B C N M']): The original tensor for reference.
        appeared_mask (Bool[Tensor, 'N M'], optional): A mask indicating which values have appeared. Defaults to None.

    Returns:
        Float[Tensor, 'B C N M']: The interpolated tensor.
        Bool[Tensor, 'N M']: A mask indicating which positions were interpolated.
        
    NOTE: This interpolation method is fully vectorized, meaning that it computes the nearest neighbors
    and interpolation weights for the entire grid in a single pass, and then updates all missing positions
    at once. This breaks the sequential dependency: newly interpolated values are not reused within the
    same pass. As a result, the method is faster than dynamic methods (interpolation_dynamic), but the results may differ.
    """
    assert x.dim() == 4, "Input tensor x should have shape (B, C, N, M)"
    assert x.shape == ori_x.shape
    batch_size, channels, N, M = x.shape 
    device = x.device
    appeared_mask = appeared_mask.squeeze().bool() if appeared_mask is not None else torch.zeros((N, M), dtype=torch.bool, device=device)
    mask = (x[:, 0] == 0)  # Shape: (batch_size, N, M)

    for b in range(batch_size):
        zero_positions = (x[b, 0] == 0)
        ref = torch.where(appeared_mask.unsqueeze(0), ori_x[b], x[b])  # (C,N,M)
        nz = ref[0] != 0  

        idx = torch.arange(M, device=device).view(1, M).expand(N, M)
        left_idx = torch.where(nz, idx, -1)
        left_idx = torch.cummax(left_idx, dim=1).values
        right_idx = torch.where(nz, idx, -1)
        right_idx = torch.flip(torch.cummax(torch.flip(right_idx, [1]), dim=1).values, [1])
        right_idx = torch.where(right_idx < 0, -1, right_idx)

        idy = torch.arange(N, device=device).view(N, 1).expand(N, M)
        up_idx = torch.where(nz, idy, -1)
        up_idx = torch.cummax(up_idx, dim=0).values
        down_idx = torch.where(nz, idy, -1)
        down_idx = torch.flip(torch.cummax(torch.flip(down_idx, [0]), dim=0).values, [0])
        down_idx = torch.where(down_idx < 0, -1, down_idx)

        left_val  = ref[:, torch.arange(N)[:, None], left_idx.clamp(0)]
        right_val = ref[:, torch.arange(N)[:, None], right_idx.clamp(0)]
        up_val    = ref[:, up_idx.clamp(0), torch.arange(M)]
        down_val  = ref[:, down_idx.clamp(0), torch.arange(M)]

        j = idx
        i = idy
        wl = 1.0 / (j - left_idx).clamp(min=0).float()
        wr = 1.0 / (right_idx - j).clamp(min=0).float()
        wu = 1.0 / (i - up_idx).clamp(min=0).float()
        wd = 1.0 / (down_idx - i).clamp(min=0).float()
        wl = torch.nan_to_num(wl, nan=0.0, posinf=0.0, neginf=0.0)
        wr = torch.nan_to_num(wr, nan=0.0, posinf=0.0, neginf=0.0)
        wu = torch.nan_to_num(wu, nan=0.0, posinf=0.0, neginf=0.0)
        wd = torch.nan_to_num(wd, nan=0.0, posinf=0.0, neginf=0.0)

        num = wl.unsqueeze(0) * left_val + wr.unsqueeze(0) * right_val + \
              wu.unsqueeze(0) * up_val   + wd.unsqueeze(0) * down_val
        W = wl + wr + wu + wd

        interp = (num / W.clamp(min=1e-6).unsqueeze(0))
        x[b, :, zero_positions] = interp[:, zero_positions].to(x.dtype)


    return x, mask

INTERPOLATION_FUNCS = {
    "dynamic": interpolation_dynamic,
    "static": interpolation_static,
}

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
# = = = = = = = = = = = Latent Relocation = = = = = = = = = = = #
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
from tqdm import tqdm
def judge_edge(point_tuple,invert_code_d):
    y,x = point_tuple[0],point_tuple[1]
    max_y,max_x = invert_code_d.shape[2],invert_code_d.shape[3]
    y = 0 if y<0 else y
    x = 0 if x<0 else x
    y = int(max_y-1) if y>max_y-1 else y
    x = int(max_x-1) if x>max_x-1 else x
    new_point_tuple = (y,x)
    return new_point_tuple

def latent_relocation(
    depth_predictor,
    original_img_np: np.ndarray,
    invert_code: torch.Tensor,
    handle_points: torch.Tensor,
    target_points: torch.Tensor,
    mask: torch.Tensor,
    fill_mode='interpolation',
    interpolation_model='static',
    relocation_model='first-win',
    gamma_ratio=1.0,
    upper_scale=5,
    lower_scale=1e-5,
    alpha=1.0,
    beta=1.0,
    scale=1.0,
    tau=1.0,
    appeared_mask=None,
):
    """Compute the updated latent code based on user-defined point drags.
    
    Args:
        depth_predictor: The depth prediction model. In our experiments, we use Depth Anything V2.
        original_img_np (Float[np.ndarray, 'H W C']): The original image in numpy array format.
        invert_code (Float[torch.Tensor, '1 C H W']): The inverted latent code.
        handle_points (Float[torch.Tensor, 'N 2']): List of handle points.
        target_points (Float[torch.Tensor, 'N 2']): List of target points.
        mask (Bool[torch.Tensor, '1 C H W']): The mask indicating the region of interest.
        fill_mode (str): The fill mode for the mask region. Defaults to 'interpolation'.
        interpolation_model (str): The interpolation model to use. Defaults to 'static'.
        relocation_model (str): The relocation model to use. Defaults to 'first-win'.
        gamma_ratio (float): The gamma ratio for displacement field computation. Defaults to 1.0.
        upper_scale (float): The upper scale for geometry-aware influence function. Defaults to 5.
        lower_scale (float): The lower scale for geometry-aware influence function. Defaults to 1e-5.
        alpha (float): The alpha parameter for geometry-aware influence function. Defaults to 1.0.
        beta (float): The beta parameter for plane-aware influence function. Defaults to 1.0.
        scale (float): The overall scale for the displacement field. Defaults to 1.0. 
        tau (float): The temperature parameter for softmax-merge relocation model. Defaults to 1.0.
        appeared_mask (Bool[torch.Tensor, '1 C H W']): The mask indicating which values have appeared. Defaults to None.
    
    Returns:
        Float[torch.Tensor, '1 C H W']: The updated latent code after relocation.
        Bool[torch.Tensor, '1 C H W']: The interpolation mask indicating which positions were interpolated.

    """ 
    
    # displacement field computation
    DF = displacement_field_computation(
        depth_predictor,
        original_img_np,
        mask, 
        handle_points, 
        target_points,
        gamma_ratio=gamma_ratio,
        upper_scale=upper_scale,
        lower_scale=lower_scale,
        alpha=alpha,
        beta=beta,
        scale=scale
    )
    
    # latent relocation
    invert_code_d = copy.deepcopy(invert_code)
    if fill_mode == 'ori':
        pass
    elif fill_mode in ['0', 'interpolation']:
        invert_code_d[(mask > 0).repeat(1,4,1,1)] = 0
    elif fill_mode == "random":
        invert_code_d[(mask > 0).repeat(1,4,1,1)] = torch.rand_like(invert_code_d)[(mask > 0).repeat(1,4,1,1)].to(device=invert_code_d.device)
    else:
        raise NotImplementedError(f"fill_mode {fill_mode} not supported")

    index_1 = torch.nonzero(mask) 
    C = index_1[:, -2:]
    move_vector = DF[C[:, 0], C[:, 1]]   
    move_vector = move_vector.flip(-1)    
    point_new = torch.round(C + move_vector).long()   
    H, W = invert_code.shape[-2:]
    point_new[:, 0] = point_new[:, 0].clamp(0, H-1)
    point_new[:, 1] = point_new[:, 1].clamp(0, W-1)
    
    # relocation model
    # NOTE: first-win is the default model used in GeoDrag.
    if relocation_model in ['first-win', 'last-win']:
        idx_dst = (point_new[:, 0] * W + point_new[:, 1])
        sorted_idx = torch.argsort(idx_dst, stable=True) if relocation_model == 'first-win'\
            else torch.argsort(idx_dst, stable=True, descending=True)
            
        idx_sorted = idx_dst[sorted_idx]
        
        keep_sorted = torch.ones_like(idx_sorted, dtype=torch.bool)
        keep_sorted[1:] = idx_sorted[1:] != idx_sorted[:-1]
        keep = torch.zeros_like(keep_sorted)
        
        keep[sorted_idx] = keep_sorted
        point_new = point_new[keep]
        C = C[keep]
        invert_code_d[:, :, point_new[:, 0], point_new[:, 1]] = invert_code[:, :, C[:, 0], C[:, 1]] 
    elif relocation_model == 'mean-merge':
        idx_src = (C[:, 0] * W + C[:, 1]).long()                 # [N]
        idx_dst = (point_new[:, 0] * W + point_new[:, 1]).long()  # [N]
        B, Cc, H, W = invert_code.shape
        vals = invert_code.view(B, Cc, H * W)[:, :, idx_src]
        vals_bc = vals.reshape(B * Cc, -1)
        
        sum_bc = torch.zeros(B * Cc, H * W, device=vals.device, dtype=vals.dtype)
        idx_dst_row = idx_dst.unsqueeze(0).expand(B * Cc, -1)  # [B*Cc, N]
        sum_bc.scatter_add_(1, idx_dst_row, vals_bc)
        
        count = torch.zeros(H * W, device=vals.device, dtype=vals.dtype)
        count.scatter_add_(0, idx_dst, torch.ones_like(idx_dst, dtype=vals.dtype))
        
        idx_unique = torch.unique(idx_dst)
        
        avg_bc = sum_bc[:, idx_unique] / count[idx_unique].clamp_min(1)
        out = invert_code_d.view(B * Cc, H * W)
        out[:, idx_unique] = avg_bc
        invert_code_d = out.view(B, Cc, H, W)
    elif relocation_model == 'softmax-merge':
        idx_src = (C[:, 0] * W + C[:, 1]).long()                 # [N]
        idx_dst = (point_new[:, 0] * W + point_new[:, 1]).long()  # [N]
        B, Cc, H, W = invert_code.shape
        
        disp = (point_new - C).float()                 # [N,2]
        score = - disp.norm(dim=1) / tau
        
        neg_inf = torch.finfo(score.dtype).min
        group_max = torch.full((H * W,), neg_inf, device=score.device, dtype=score.dtype)
        group_max.scatter_reduce_(0, idx_dst, score, reduce="amax", include_self=True)  # [H*W]
        score_centered = score - group_max[idx_dst]  # [N]
        
        w = torch.exp(score_centered).to(invert_code.dtype)       # [N]
        
        vals = invert_code.view(B, Cc, H * W)[:, :, idx_src]      # [B, Cc, N]
        vals_bc = (vals * w).reshape(B * Cc, -1)                  # [B*Cc, N]

        sum_bc = torch.zeros(B * Cc, H * W, device=vals.device, dtype=vals.dtype)
        sum_bc.scatter_add_(1, idx_dst.unsqueeze(0).expand(B * Cc, -1), vals_bc)

        w_sum = torch.zeros(H * W, device=vals.device, dtype=vals.dtype)
        w_sum.scatter_add_(0, idx_dst, w)
        
        idx_unique = torch.unique(idx_dst)
        out = invert_code_d.view(B * Cc, H * W)
        out[:, idx_unique] = sum_bc[:, idx_unique] / w_sum[idx_unique].clamp_min(1e-6)
        invert_code_d = out.view(B, Cc, H, W)
    elif relocation_model == 'overwrite':
        invert_code_d[:, :, point_new[:, 0], point_new[:, 1]] = invert_code[:, :, C[:, 0], C[:, 1]]
        
    invert_code_d, interpolation_mask = INTERPOLATION_FUNCS[interpolation_model](invert_code_d, invert_code, appeared_mask=appeared_mask)
    
    return invert_code_d, interpolation_mask

def run_drag(
    diffusion_model,
    depth_predictor,
    model_input,
    configs,
    lora_path,
    *,
    device=torch.device('cuda'),
    dtype=torch.float16,
):  
    """Run the drag operation.

    Args:
        diffusion_model: The diffusion model to use.
        depth_predictor: The depth prediction model.
        model_input (dict): The input dictionary containing:
            - 'original_image' (Float[torch.Tensor, '1 C H W']): The original image tensor.
            - 'prompt' (List[str]): The text prompt.
            - 'mask' (Bool[torch.Tensor, '1 1 H W']): The mask tensor.
            - 'handle_points' (Float[torch.Tensor, 'N 2']): The handle points tensor.
            - 'target_points' (Float[torch.Tensor, 'N 2']): The target points tensor.
            - 'image' (Float[np.ndarray, 'H W C']): The original image in numpy array format.
        configs (dict): The configuration dictionary containing:
            - 'n_inference_step' (int): Number of inference steps.
            - 'n_actual_inference_step' (int): Number of actual inference steps.
            - 'guidance_scale' (float): The guidance scale.
            - 'eta' (float): The eta parameter for DDIM.
            - 'gamma' (float): The gamma ratio for displacement field computation.
            - 'upper_scale' (float): The upper scale for geometry-aware influence function.
            - 'lower_scale' (float): The lower scale for geometry-aware influence function.
            - 'alpha' (float): The alpha parameter for geometry-aware influence function.
            - 'beta' (float): The beta parameter for plane-aware influence function.
            - 'points_scale' (float): The overall scale for the displacement field.
            - 'fill_mode' (str): The fill mode for the mask region.
            - 'interpolation_model' (str): The interpolation model to use.
        lora_path (str): The path to the LoRA weights. If None, no LoRA is applied.
        device (torch.device, optional): The device to use. Defaults to torch.device('cuda').
        dtype (torch.dtype, optional): The data type to use. Defaults to torch.float16.
    
    Returns:
        Float[torch.Tensor, '1 C H W']: The generated image tensor after drag operation
    """
    if lora_path is not None and lora_path != '':
        print("applying lora: " + lora_path)
        diffusion_model.unet.load_attn_procs(lora_path)
    
    original_images = model_input['original_image'].to(device, dtype)
    text_embeddings = diffusion_model.get_text_embeddings(model_input['prompt']).to(device)
    mask = model_input['mask'].to(device)
    handle_points = model_input['handle_points'].to(device, dtype)
    target_points = model_input['target_points'].to(device, dtype)
    original_img_np = model_input['image']
    prompt = model_input['prompt']

    # ddim inversion with kv cache
    # In the inversion stage, we use the same number of steps as in the generation stage.
    # The key and value caches are stored in the attenion processors (see pipelines/modules/attention_processor.py).
    invert_code = diffusion_model.invert(
        original_images,
        prompt,
        text_embeddings=text_embeddings,
        guidance_scale=configs['guidance_scale'],
        num_inference_steps=configs['n_inference_step'],
        num_actual_inference_steps=configs['n_actual_inference_step'],)
    
    # latent relocation with geometry and spatial plane guidance
    updated_code, interpolation_mask = latent_relocation(
        depth_predictor,
        original_img_np,
        invert_code = invert_code,
        handle_points = handle_points,
        target_points = target_points,
        mask = mask,
        gamma_ratio = configs['gamma'],
        upper_scale = configs['upper_scale'],
        lower_scale = configs['lower_scale'],
        alpha = configs['alpha'],
        beta = configs['beta'],
        scale = configs['points_scale'],
        tau = configs['tau'],
        fill_mode = configs['fill_mode'],
        interpolation_model = configs['interpolation_model'],
        relocation_model = configs['relocation_model'],
    )

    # generate edited image
    torch.cuda.empty_cache()
    
    diffusion_model.scheduler.set_timesteps(configs['n_inference_step'])
    
    text_embeddings = text_embeddings.half()
    diffusion_model.unet = diffusion_model.unet.half()
    updated_code = updated_code.half()
    invert_code = invert_code.half()
    
    gen_image = diffusion_model(
        prompt=prompt,
        text_embeddings=text_embeddings,
        batch_size=1,
        latents=updated_code,
        guidance_scale=configs['guidance_scale'],
        num_inference_steps=configs['n_inference_step'],
        num_actual_inference_steps=configs['n_actual_inference_step'],
        eta=configs['eta'],
        mask=interpolation_mask
    )
    
    return gen_image
    