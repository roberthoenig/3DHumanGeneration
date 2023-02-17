import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np
from code.GraphAE.positions_dataset import PositionsDataset
from utils import graphAE_param as Param
from DataLoader import graphAE_dataloader as Dataloader
from plyfile import PlyData
from utils.utils import make_beta_schedule, extract, get_faces_from_ply, p_sample_loop, parallel_to_cpu_state_dict, ply_to_png, img_folder_to_np
from Models import conditional_model as conditional_model, graphAE as graphAE, EMA as EMA
from PIL import Image
import os
from torchvision import transforms


n_steps = 100  # number of steps
num_steps = n_steps
# betas = make_beta_schedule(schedule='linear', n_timesteps=num_steps, start=1e-3, end=1e-3)
# betas = make_beta_schedule(schedule='linear', n_timesteps=num_steps, start=1e-3, end=1e-3)
betas = make_beta_schedule(schedule='sigmoid', n_timesteps=n_steps, start=1e-5, end=1e-2)

# betas = make_beta_schedule(schedule='sigmoid', n_timesteps=num_steps, start=1e-5, end=1e-2)
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar = alphas_prod
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

param_mesh = Param.Parameters()
param_mesh.read_config("../../train/0422_graphAE_dfaust/30_conv_pool.config")

# param.augmented_data=True
param_mesh.batch = 1

param_mesh.read_weight_path = "../../train/0422_graphAE_dfaust/weight_30/model_epoch0018.weight"

test_npy_fn = "../../data/DFAUST/train.npy"

out_test_folder = "../../train/0422_graphAE_dfaust/diffusion/"

out_ply_folder = out_test_folder + "ply/"

pc_lst = np.load(test_npy_fn)
avg_height = pc_lst[:, :, 1].mean(1)

dir_path = os.path.dirname(os.path.realpath(__file__))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Pad(int((256 - 224) / 2)),
])
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
dataset = PositionsDataset(csv_file="../../data/positions/labels.csv", root_dir="../../data/positions/scene_silhouette_imgs", transform=preprocess)
dloader = Dataloader(dataset, batch_size=1, shuffle=True, num_workers=0)

if __name__ == "__main__":
    model = conditional_model.ConditionalModelPositionFromRGBScenesAndSilhouettes(
        n_steps, in_sz=3, cond_sz_scene=64, cond_sz_silhouette=64, dsize=len(dataset), device='cpu')
    diffusion_model = conditional_model.ConditionalModel(n_steps, in_sz=63, cond_sz=64, cond_model=True)
    state_dict = torch.load("../../train/0422_graphAE_dfaust/position/cond_model_3999.pt")
    state_dict = parallel_to_cpu_state_dict(state_dict)
    diffusion_model.load_state_dict(state_dict)
    diffusion_model.eval()

    optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-3)
    # Create EMA model
    ema = EMA.EMA(0.9)
    ema.register(diffusion_model)

    mesh_model = graphAE.Model(param_mesh, test_mode=True)
    mesh_model.cuda()

    if param_mesh.read_weight_path != "":
        checkpoint = torch.load(param_mesh.read_weight_path)
        mesh_model.load_state_dict(checkpoint['model_state_dict'])
        mesh_model.init_test_mode()

    mesh_model.eval()

    template_plydata = PlyData.read(param_mesh.template_ply_fn)
    faces = get_faces_from_ply(template_plydata)

    for j in range(3):
        for i, cond in tqdm(list(enumerate(cond_dataset))):
            sample = p_sample_loop(n_steps, diffusion_model, [1, 63], alphas, one_minus_alphas_bar_sqrt, betas, cond.unsqueeze(0))
            sample = np.concatenate([s.detach().numpy() for s in sample])
            sample = sample.reshape(-1, 7, 9)

            sample_torch = torch.FloatTensor(sample).cuda()

            mesh = sample_torch[-1]
            mesh = torch.unsqueeze(mesh, dim=0)
            # print(mesh.shape)
            out_mesh = mesh_model.forward_from_layer_n(mesh, 8)
            out_mesh = out_mesh.cpu()
            # print(out_mesh.shape)
            pc_out = np.array(out_mesh[0].data.tolist())
            pc_out[:, 1] += avg_height[0]
            fname = "%08d" % (i) + "_out_" + str(j)
            Dataloader.save_pc_into_ply(template_plydata, pc_out, out_ply_folder + fname+".ply")
            ply_to_png(out_ply_folder + fname+".ply", out_ply_folder + fname + "_rendering" + ".png", silhouette=False)
            v = cond.squeeze().numpy()
            im = Image.fromarray(np.uint8(v*255), 'L')
            im.save(out_ply_folder + fname + ".png")
