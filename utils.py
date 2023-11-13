import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import find_boundaries


def compute_data_for_superpixel_boundary_recall(sp_map_np, gt_np):
    sp_boundary_mask = find_boundaries(sp_map_np, mode='inner', background=-1)
    gt_boundary_mask = find_boundaries(gt_np, mode='inner')
    tp = np.sum(np.logical_and(sp_boundary_mask, gt_boundary_mask))
    fn = np.sum(np.logical_and(sp_boundary_mask == 0, gt_boundary_mask))

    return tp, fn


def conv_label_to_one_hot_tensor(labels, c=2):
    b, _, h, w = labels.shape
    one_hot = torch.zeros(b, c, h, w, dtype=torch.long, device=labels.device)
    one_hot.scatter_(1, labels.long(), 1)

    return one_hot.float()


def generate_target_tensor(label):
    label_one_hot = conv_label_to_one_hot_tensor(label)

    b, _, h, w = label.shape
    coords_y, coords_x = torch.meshgrid(torch.arange(h, device=label.device),
                                        torch.arange(w, device=label.device))
    coords = torch.stack([coords_x, coords_y])
    coords = torch.tile(coords, (b, 1, 1, 1))

    target = torch.cat([label_one_hot, coords], dim=1)

    return target


def neighborhood_values(np_arr: np.ndarray, mh=1, mw=1):
    h, w = np_arr.shape
    arr_pd = np.pad(np_arr, np.array([[mh, mh], [mw, mw]]), mode='edge')

    shifted_arrays = []
    for hi in range(mh * 2 + 1):
        for wi in range(mw * 2 + 1):
            shifted_arrays.append(arr_pd[hi: hi + h, wi: wi + w])

    result = np.dstack(shifted_arrays)
    return result


def update_superpixel_map(sp_index, assign_map):
    _, assign_max_idx = torch.max(assign_map, dim=1, keepdim=True)
    superpixel_map = torch.gather(sp_index, dim=1, index=assign_max_idx).long()

    return superpixel_map


def get_superpixel_map(image_h, image_w, downsize, batch_size, assign_map, device):
    spixel_index = init_spixel_index(image_h, image_w, downsize, batch_size, device)
    superpixel_map = update_superpixel_map(spixel_index, assign_map)

    return superpixel_map


def init_spixel_index(ph, pw, downsize, b, device):
    sph = ph // downsize
    spw = pw // downsize

    sp_idx_map = np.arange(sph * spw).reshape((sph, spw))
    sp_neighbour_idx_map = neighborhood_values(sp_idx_map)

    sp_neighbour_idx_map_tensor = torch.from_numpy(sp_neighbour_idx_map).permute(2, 0, 1).to(device=device)

    sp_idx_tensor = torch.repeat_interleave(
                    torch.repeat_interleave(sp_neighbour_idx_map_tensor, downsize, dim=1),
                    downsize, dim=2)
    sp_idx_tensor = torch.unsqueeze(sp_idx_tensor, 0).float()
    sp_idx_tensor = F.interpolate(sp_idx_tensor, size=(ph, pw), mode='nearest')
    sp_idx_tensor = torch.tile(sp_idx_tensor, (b, 1, 1, 1))

    return sp_idx_tensor.long()


def feature_p2sp(pixel_feat, assign_q, sp_idx):
    # pixel_feat: [b, c, hw]
    # assign_q  : [b, 9, hw]
    # sp_idx    : [b, 9, hw]

    num_sp = torch.max(sp_idx).item() + 1
    b, c, num_p = pixel_feat.shape

    feat_ = torch.cat([pixel_feat, torch.ones([b, 1, num_p], device=pixel_feat.device)], dim=1)
    c1 = c + 1
    sp_feat_ = torch.zeros([b, c1, num_sp], device=pixel_feat.device)

    # 0
    ff = feat_ * assign_q.narrow(1, 0, 1)
    sp_feat_.scatter_add_(dim=2, index=sp_idx.narrow(1, 0, 1).expand(b, c1, num_p), src=ff)

    # 1
    ff = feat_ * assign_q.narrow(1, 1, 1)
    sp_feat_.scatter_add_(dim=2, index=sp_idx.narrow(1, 1, 1).expand(b, c1, num_p), src=ff)

    # 2
    ff = feat_ * assign_q.narrow(1, 2, 1)
    sp_feat_.scatter_add_(dim=2, index=sp_idx.narrow(1, 2, 1).expand(b, c1, num_p), src=ff)

    # 3
    ff = feat_ * assign_q.narrow(1, 3, 1)
    sp_feat_.scatter_add_(dim=2, index=sp_idx.narrow(1, 3, 1).expand(b, c1, num_p), src=ff)

    # 4
    ff = feat_ * assign_q.narrow(1, 4, 1)
    sp_feat_.scatter_add_(dim=2, index=sp_idx.narrow(1, 4, 1).expand(b, c1, num_p), src=ff)

    # 5
    ff = feat_ * assign_q.narrow(1, 5, 1)
    sp_feat_.scatter_add_(dim=2, index=sp_idx.narrow(1, 5, 1).expand(b, c1, num_p), src=ff)

    # 6
    ff = feat_ * assign_q.narrow(1, 6, 1)
    sp_feat_.scatter_add_(dim=2, index=sp_idx.narrow(1, 6, 1).expand(b, c1, num_p), src=ff)

    # 7
    ff = feat_ * assign_q.narrow(1, 7, 1)
    sp_feat_.scatter_add_(dim=2, index=sp_idx.narrow(1, 7, 1).expand(b, c1, num_p), src=ff)

    # 8
    ff = feat_ * assign_q.narrow(1, 8, 1)
    sp_feat_.scatter_add_(dim=2, index=sp_idx.narrow(1, 8, 1).expand(b, c1, num_p), src=ff)

    sp_f = sp_feat_[:, :-1, :] / (sp_feat_[:, -1:, :] + 1e-8)
    return sp_f


def feature_sp2p(spixel_feat, assign_q, sp_idx):
    # spixel_feat: [b, c, nsp]
    # assign_q   : [b, 9, hw]
    # sp_idx     : [b, 9, hw]

    b, c = spixel_feat.shape[:2]
    num_p = assign_q.shape[-1]

    # 0
    p_feat = torch.gather(spixel_feat, dim=2, index=sp_idx.narrow(1, 0, 1).expand(b, c, num_p))
    p_feat = p_feat * assign_q.narrow(1, 0, 1)

    # 1
    ff = torch.gather(spixel_feat, dim=2, index=sp_idx.narrow(1, 1, 1).expand(b, c, num_p))
    p_feat += ff * assign_q.narrow(1, 1, 1)

    # 2
    ff = torch.gather(spixel_feat, dim=2, index=sp_idx.narrow(1, 2, 1).expand(b, c, num_p))
    p_feat += ff * assign_q.narrow(1, 2, 1)

    # 3
    ff = torch.gather(spixel_feat, dim=2, index=sp_idx.narrow(1, 3, 1).expand(b, c, num_p))
    p_feat += ff * assign_q.narrow(1, 3, 1)

    # 4
    ff = torch.gather(spixel_feat, dim=2, index=sp_idx.narrow(1, 4, 1).expand(b, c, num_p))
    p_feat += ff * assign_q.narrow(1, 4, 1)

    # 5
    ff = torch.gather(spixel_feat, dim=2, index=sp_idx.narrow(1, 5, 1).expand(b, c, num_p))
    p_feat += ff * assign_q.narrow(1, 5, 1)

    # 6
    ff = torch.gather(spixel_feat, dim=2, index=sp_idx.narrow(1, 6, 1).expand(b, c, num_p))
    p_feat += ff * assign_q.narrow(1, 6, 1)

    # 7
    ff = torch.gather(spixel_feat, dim=2, index=sp_idx.narrow(1, 7, 1).expand(b, c, num_p))
    p_feat += ff * assign_q.narrow(1, 7, 1)

    # 8
    ff = torch.gather(spixel_feat, dim=2, index=sp_idx.narrow(1, 8, 1).expand(b, c, num_p))
    p_feat += ff * assign_q.narrow(1, 8, 1)

    return p_feat


def feature_reconstruction(assign_q, feat, sp_idx):
    # assign_q: [b, 9, hw]
    # feat    : [b, c, hw]
    # sp_idx  : [b, 9, hw]

    sp_feat = feature_p2sp(feat, assign_q, sp_idx)
    reconstructed_feat = feature_sp2p(sp_feat, assign_q, sp_idx)

    return reconstructed_feat


def compute_loss_sp(assign_q, target_feat, downsize, pos_weight):
    b, _, h, w = assign_q.shape
    device = assign_q.device

    sp_idx = init_spixel_index(h, w, downsize, b, device).long()

    assign_q = torch.flatten(assign_q, start_dim=2)
    target_feat = torch.flatten(target_feat, start_dim=2)
    sp_idx = torch.flatten(sp_idx, start_dim=2)

    reconstructed_feat = feature_reconstruction(assign_q, target_feat, sp_idx)

    logits = torch.log(reconstructed_feat[:, :-2] + 1e-8)
    loss_sp_sem = -torch.sum(logits * target_feat[:, :-2]) / b / 1000

    loss_sp_pos = F.mse_loss(reconstructed_feat[:, -2:], target_feat[:, -2:])

    loss_sp = loss_sp_sem + pos_weight * loss_sp_pos

    return loss_sp, loss_sp_sem, loss_sp_pos


def compute_loss_se(pred, label):
    label = label.long().squeeze(dim=1)
    return F.cross_entropy(pred, label)
