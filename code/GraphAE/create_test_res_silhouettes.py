import torch
import numpy as np
from Models import graphAE as graphAE
from utils import graphAE_param as Param
from DataLoader import graphAE_dataloader as Dataloader
from plyfile import PlyData
from utils.utils import get_faces_from_ply, get_colors_from_diff_pc, dump
import os
import open3d as o3d
import trimesh
import pyrender
from PIL import Image
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import natsort
from PIL import Image


def ply_to_png(ply_filename, png_filename, silhouette=False):
    mesh = trimesh.load(ply_filename)
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
    camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
    scene.add(mesh, pose=  np.eye(4))
    scene.add(light, pose=  np.eye(4))
    scene.add(camera, pose=[[ 1,  0,  0,  0],
                            [ 0,  1, 0, 0],
                            [ 0,  0,  1,  2],
                            [ 0,  0,  0,  1]])
    # render scene
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)
    if silhouette:
        color = (255 * (color > 0)).astype(np.uint8)
    img = Image.fromarray(color)
    img.save(png_filename)

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
    print("pc_lst.shape", pc_lst.shape)
    print(pc_lst.shape[0], "meshes in total.")

    geo_error_sum = 0
    laplace_error_sum = 0
    pc_num = len(pc_lst)
    out = np.empty([len(pc_lst), 32, 7, 9])
    data = np.load("../../data/DFAUST/test_res.npy")
    indices = list(range(0, 32928, 32))
    data = data[indices, :, :, :]
    dataset = torch.Tensor(data).float()
    n = 0
    for batch_idx in tqdm(range(dataset.shape[0])):
        batch = dataset[batch_idx]
        if param.generate_encoded_data:
            out_pcs_torch = batch.cuda()
            
            out_mesh = model.forward_from_layer_n(out_pcs_torch, 8)
            out_mesh = out_mesh.cpu()
            for i in range(out_mesh.shape[0]):
                pc_out = np.array(out_mesh[i].data.tolist())
                Dataloader.save_pc_into_ply(template_plydata, pc_out, "_tmp_out.ply")
                ply_to_png("_tmp_out.ply", f"silhouettes/{str(n).zfill(5)}_silhouette.png", silhouette=True)
                # ply_to_png("_tmp_out.ply", f"silhouettes/{n}.png", silhouette=False)
                n = n+1

if __name__ == "__main__":
    create_pngs = True
    if create_pngs:
        param = Param.Parameters()
        param.read_config("../../train/0422_graphAE_dfaust/30_conv_pool.config")
        param.generate_encoded_data = True

        # param.augmented_data=True
        param.batch = 32

        param.read_weight_path = "../../train/0422_graphAE_dfaust/weight_30/model_epoch0018.weight"
        print(param.read_weight_path)

        test_npy_fn = "../../data/DFAUST/train.npy"

        out_test_folder = "../../train/0422_graphAE_dfaust/test_30/epoch18/"

        out_ply_folder = out_test_folder + "ply/"

        if not os.path.exists(out_ply_folder):
            os.makedirs(out_ply_folder)

        pc_lst = np.load(test_npy_fn)

        with torch.no_grad():
            torch.manual_seed(10)
            np.random.seed(10)

            test(param, test_npy_fn, out_ply_folder, skip_frames=0)

    path = "/home/robert/g/3DHumanGeneration/code/GraphAE/silhouettes/"
    img_filenames = [f for f in listdir(path) if isfile(join(path, f))]
    img_filenames = natsort.natsorted(img_filenames,reverse=False)
    arrays = []
    for img_filename in tqdm(img_filenames):
        with Image.open(path + img_filename) as img:
            img = img.resize((224, 224))
            arr = np.array(img)
            arr = arr[:,:,0] == 255
            arrays.append(arr)
    save_path = "/home/robert/g/3DHumanGeneration/data/DFAUST/test_res_silhouettes.npy"
    silhouette_dataset = np.stack(arrays, axis=0)
    print(silhouette_dataset.shape)
    np.save(save_path, silhouette_dataset)