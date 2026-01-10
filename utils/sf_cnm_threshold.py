# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : 2026/1/9 20:17   (Revised)
# @Function   : Get_Mis_Threshold; Compute_Attention_Label
# @Description: 
#              1. Get_Mis_Threshold : 根据光谱样本矩阵计算误分类阈值
#                  ->input  : 光谱矩阵(numpy格式多维张量[n,n])
#                  ->output : 阈值向量(torch格式张量[1,n])
#
#              2. Compute_Attention_Label : 计算待识别影像的每个像素位置中的波谱与光谱样本的相似距离,同时根据阈值排除不属于光谱样本中的像素;把相似的距离转化为0-1之间的相似度，把排除的设置为0
#                  ->input   : 待识别影像(numpy格式多维张量[b,c,h,w]), 光谱矩阵(numpy格式多维张量[n,n]), 阈值向量(torch格式张量[1,n])
#                  ->output  : 相似度图(torch格式张量[b,1,h,w])


import torch
import numpy as np
import torch.nn.functional as F



def Get_Mis_Threshold (Matrix):
    ill_cond_A        = torch.tensor(Matrix, dtype=torch.float32)
    rows, columns     = ill_cond_A.shape
    Lambda            = rows + 1
    well_cond_A       = ill_cond_A + Lambda * torch.eye(rows)
    well_cond_A_inv   = torch.linalg.inv(well_cond_A)
    ill_cond_A_cols   = ill_cond_A.T
    lambda_vectors    = Lambda * torch.eye(columns)
    perturbed_vectors = ill_cond_A_cols.unsqueeze(1) + lambda_vectors.unsqueeze(0)
    results           = torch.matmul(well_cond_A_inv, perturbed_vectors.unsqueeze(-1)).squeeze(-1)
    result_j          = results.gather(dim=2, index=torch.arange(columns).repeat(rows, 1).unsqueeze(-1)).squeeze(-1)
    abs_arr           = torch.abs(results)
    sum_abs           = abs_arr.sum(dim=2)
    sum_rest          = sum_abs - torch.abs(result_j)
    Threshold_Matrix  = torch.abs(result_j - 1) + sum_rest

    masked_A          = Threshold_Matrix.clone()
    masked_A[torch.arange(rows), torch.arange(columns)] = float('inf')
    mis_threshold     = torch.min(masked_A, dim=0).values

    return mis_threshold




def Compute_Attention_Label(img, Matrix, threshold):
    B,C,H,W           = img.size()
    lambdas           = C+1
    pixels            = img.permute(0, 2, 3, 1).reshape(-1, C)              # [B*H*W, C]
    lambda_perturb    = lambdas * torch.eye(C, device=img.device)           # [C, C]
    perturbed_input   = pixels.unsqueeze(1) + lambda_perturb.unsqueeze(0)   # [B*H*W, C, C]
    perturbed_input   = perturbed_input.to(img.device)

    ill_cond_A        = torch.tensor(Matrix, dtype=torch.float32)           # [C, C]
    rows, columns     = ill_cond_A.shape
    Lambda            = rows + 1
    well_cond_A       = ill_cond_A + Lambda * torch.eye(rows)               # [C, C]
    well_cond_A_inv   = torch.linalg.inv(well_cond_A)                       # [C, C]
    outputs           = torch.matmul(well_cond_A_inv, perturbed_input.unsqueeze(-1)).squeeze(-1) # [B*H*W, C, C]

    # Manhattan Distance
    abs_outputs       = outputs.abs()
    max_vals, _       = abs_outputs.max(dim=2)
    sum_rest          = abs_outputs.sum(dim=2) - max_vals
    distances         = sum_rest + (max_vals - 1).abs()

    # Similarity Calculation after threshold elimination
    min_dist, min_idx = torch.min(distances, dim=1)
    valid_mask        = min_dist < threshold[min_idx]
    min_dist_3d       = min_dist.view(B, H, W)           # [B, H, W]
    valid_mask_3d     = valid_mask.view(B, H, W)         # [B, H, W]
    similar_lbl       = torch.zeros((B, H, W), device=min_dist.device)

    for b in range(B):
        valid_vals    = min_dist_3d[b][valid_mask_3d[b]]

        if valid_vals.numel() == 0:
            continue

        min_val                          = valid_vals.min()
        max_val                          = valid_vals.max()
        inv_norm                         = 1.0 - (valid_vals - min_val) / (max_val - min_val + 1e-8)
        similar_lbl[b][valid_mask_3d[b]] = inv_norm

    similar_lbl = similar_lbl.unsqueeze(1)    # [B, 1, H, W]
#    similar_128 = F.interpolate(similar_lbl, size=(128, 128), mode='area')
#    similar_64  = F.interpolate(similar_lbl, size=(64, 64),   mode='area')
#    similar_32  = F.interpolate(similar_lbl, size=(32, 32),   mode='area')

    # classify map
    class_map_3d             = min_idx.view(B, H, W)     # [B, H, W]
    label_map                = torch.zeros((B, H, W), device=img.device, dtype=torch.long)
    label_map[valid_mask_3d] = class_map_3d[valid_mask_3d] + 1

    return similar_lbl, label_map


# Test suite
if __name__ == "__main__":
    from osgeo import gdal
    gdal.UseExceptions()
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Times New Roman"
    })

    dataset    = gdal.Open('../test_sample/365.tif').ReadAsArray()
    #img_tensor = torch.Tensor(dataset/10000).unsqueeze(0) # [1, 10, 128, 128]
    img_tensor = torch.Tensor(dataset).unsqueeze(0)       # [1, 10, 128, 128]
    print(f"输入影像尺寸: {img_tensor.shape}", flush=True)

    ill_cond_A = np.array([
                [0.025108, 0.044775, 0.022713, 0.054517, 0.024255, 0.263413, 0.897558, 0.083041, 0.051523, 0.447467],
                [0.039021, 0.065726, 0.041806, 0.077207, 0.041771, 0.281572, 0.906344, 0.111226, 0.092698, 0.477236],
                [0.069077, 0.102917, 0.077322, 0.119946, 0.078582, 0.304236, 0.903852, 0.165515, 0.191769, 0.511057],
                [0.082601, 0.113206, 0.087972, 0.137899, 0.092967, 0.302216, 0.902451, 0.18202 , 0.192265, 0.536082],
                [0.111036, 0.139278, 0.096018, 0.146704, 0.098742, 0.231165, 0.737299, 0.199735, 0.150488, 0.497987],
                [0.110326, 0.150545, 0.140082, 0.20488 , 0.116332, 0.008532, 0.030921, 0.263537, 0.018009, 0.012086],
                [0.090256, 0.125565, 0.124102, 0.174439, 0.098414, 0.00814 , 0.032178, 0.232524, 0.019361, 0.01256],
                [0.608871, 0.597479, 0.526987, 0.49861 , 0.533335, 0.625007, 0.691949, 0.537595, 0.60378 , 0.654422],
                [0.119021, 0.078705, 0.108357, 0.080463, 0.103712, 0.094274, 0.06156 , 0.273285, 0.064751, 0.119046],
                [0.390516, 0.407727, 0.355867, 0.367505, 0.403623, 0.972147, 0.96713 , 0.385322, 0.894642, 0.976719]])  

#    ill_cond_A = np.array([
#        [0.1067, 0.0265, 0.2270, 0.4892, 0.0819, 0.0409, 0.0357],
#        [0.1206, 0.0176, 0.2251, 0.4855, 0.0721, 0.0463, 0.0359],
#        [0.1827, 0.0132, 0.2440, 0.5294, 0.0382, 0.0478, 0.0586],
#        [0.2375, 0.0098, 0.2256, 0.4978, 0.0081, 0.0115, 0.0278],
#        [0.3144, 0.0073, 0.1291, 0.3052, 0.0023, 0.0011, 0.0028],
#        [0.3947, 0.0015, 0.0105, 0.0375, 0.0019, 0.0011, 0.0089],
#        [0.3195, 0.0019, 0.0084, 0.0346, 0.0014, 0.0014, 0.0088]])

    print("参考光谱矩阵为(每列是某物质波谱)：\n",ill_cond_A, flush=True)
    mis_threshold = Get_Mis_Threshold(ill_cond_A)
    print("阈值计算完成！", flush=True)
    print("阈值结果为：",mis_threshold, flush=True)
    print(f"计算注意力标签中......", flush=True)
    similarity_map, classify_map = Compute_Attention_Label(img_tensor, ill_cond_A, mis_threshold)
    print(f"注意力标签尺寸为: {similarity_map.shape}", flush=True)

    # Similarity Map 可视化与保存
    import os
    os.makedirs('../figure', exist_ok=True)

    similarity_np = similarity_map.squeeze().cpu().detach().numpy()
    plt.figure(figsize=(13, 6))
    sim_img       = plt.imshow(similarity_np, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig('../figure/similarity_map_origin.tif', dpi=300, bbox_inches='tight')

    plt.colorbar(sim_img, label='Similarity Confidence (0–1)')
    plt.title('Spectral Similarity Map')
    plt.savefig('../figure/similarity_map.tif', dpi=300, bbox_inches='tight')
    plt.show()
    # Similarity Map 可视化与保存
    label_np    = classify_map.squeeze().cpu().numpy()   # [H, W]
    num_classes = int(label_np.max())
    import matplotlib.colors as mcolors
    base_colors = list(plt.cm.tab20.colors)
    colors      = ['black'] + base_colors[:num_classes]
    cmap        = mcolors.ListedColormap(colors)
    bounds      = np.arange(0, num_classes + 2) - 0.5
    norm        = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(13, 6))
    img         = plt.imshow(label_np, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.savefig('../figure/classify_map_origin.tif', dpi=300, bbox_inches='tight')

    cbar        = plt.colorbar(img, ticks=np.arange(0, num_classes + 1))
    cbar.set_label('Spectral Class Index (0=invalid)')
    plt.title('Classification Map')
    plt.savefig('../figure/classify_map.tif', dpi=300, bbox_inches='tight')
    plt.show()
