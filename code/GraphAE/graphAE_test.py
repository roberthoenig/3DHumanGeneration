# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import numpy as np
from Models import graphAE as graphAE
from utils import graphAE_param as Param
from DataLoader import graphAE_dataloader as Dataloader
from plyfile import PlyData
from utils import get_faces_from_ply, get_colors_from_diff_pc
import os


def test(param, test_npy_fn, out_ply_folder, skip_frames=0):
    print("**********Initiate Netowrk**********")
    model = graphAE.Model(param, test_mode=True)

    model.cuda()

    if (param.read_weight_path != ""):
        print("load " + param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.init_test_mode()

    model.eval()

    template_plydata = PlyData.read(param.template_ply_fn)
    faces = get_faces_from_ply(template_plydata)

    pose_sum = 0
    laplace_sum = 0
    test_num = 0

    print("**********Get test pcs**********", test_npy_fn)
    # get ply file lst
    pc_lst = np.load(test_npy_fn)
    print(pc_lst.shape[0], "meshes in total.")

    geo_error_sum = 0
    laplace_error_sum = 0
    pc_num = len(pc_lst)
    n = 0
    out = np.empty([len(pc_lst), 32, 7, 9])

    while n < (pc_num - 1):

        batch = min(pc_num - n, param.batch)
        pcs = pc_lst[n:n + batch]
        height = pcs[:, :, 1].mean(1)

        # centralize each instance
        pcs[:, :, 0:3] -= pcs[:, :, 0:3].mean(1).reshape((-1, 1, 3)).repeat(param.point_num,
                                                                            1)

        pcs_torch = torch.FloatTensor(pcs).cuda()
        if param.augmented_data:
            pcs_torch = Dataloader.get_augmented_pcs(pcs_torch)
        if batch < param.batch:
            pcs_torch = torch.cat((pcs_torch, torch.zeros(param.batch - batch, param.point_num, 3).cuda()), 0)

        if param.generate_encoded_data:
            out_pcs_torch = model.forward_till_layer_n(pcs_torch, 8)
            out_pcs_torch = out_pcs_torch.cpu()
            out[n, :, :, :] = out_pcs_torch
        else:
            out_pcs_torch = model(pcs_torch)
            geo_error = model.compute_geometric_mean_euclidean_dist_error(pcs_torch[0:batch], out_pcs_torch[0:batch])
            geo_error_sum += geo_error * batch
            laplace_error_sum = laplace_error_sum + model.compute_laplace_Mean_Euclidean_Error(pcs_torch[0:batch],
                                                                                               out_pcs_torch[
                                                                                               0:batch]) * batch
            print(n, geo_error.item())

            if n % 128 == 0:
                print(height[0])
                pc_gt = np.array(pcs_torch[0].data.tolist())
                pc_gt[:, 1] += height[0]
                pc_out = np.array(out_pcs_torch[0].data.tolist())
                pc_out[:, 1] += height[0]

                diff_pc = np.sqrt(pow(pc_gt - pc_out, 2).sum(1))
                color = get_colors_from_diff_pc(diff_pc, 0, 0.02) * 255
                Dataloader.save_pc_with_color_into_ply(template_plydata, pc_out, color,
                                                       out_ply_folder + "%08d" % (n) + "_out.ply")
                Dataloader.save_pc_into_ply(template_plydata, pc_gt, out_ply_folder + "%08d" % (n) + "_gt.ply")

        n = n + batch

    if not param.generate_encoded_data:
        geo_error_avg = geo_error_sum.item() / pc_num
        laplace_error_avg = laplace_error_sum.item() / pc_num

        print("geo error:", geo_error_avg, "laplace error:", laplace_error_avg)

    return out


param = Param.Parameters()
param.read_config("../../train/0422_graphAE_dfaust/30_conv_pool.config")
param.generate_encoded_data = True

# param.augmented_data=True
param.batch = 32

param.read_weight_path = "../../train/0422_graphAE_dfaust/weight_30/model_epoch0198.weight"
print(param.read_weight_path)

test_npy_fn = "../../data/DFAUST/train.npy"

out_test_folder = "../../train/0422_graphAE_dfaust/test_30/epoch198/"

out_ply_folder = out_test_folder + "ply/"

if not os.path.exists(out_ply_folder):
    os.makedirs(out_ply_folder)

pc_lst = np.load(test_npy_fn)

with torch.no_grad():
    torch.manual_seed(10)
    np.random.seed(10)

    outly = test(param, test_npy_fn, out_ply_folder, skip_frames=0)
    if param.generate_encoded_data:
        np.save("../../data/DFAUST/test_res.npy", outly)
