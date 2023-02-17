import math
import PIL
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import torch
from torchvision import transforms
import torchvision
import torch.optim as optim
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
from Models.EMA import EMA as EMA
from Models.conditional_model import ConditionalModel, ConditionalModelPositionFromRGBScenesAndSilhouettes
from utils.load_scene_ply import scene_to_trimeshes
from utils.utils import make_beta_schedule, noise_estimation_loss, parallel_to_cpu_state_dict, p_sample_loop
from positions_dataset import PositionsDataset
from operator import itemgetter
from DataLoader import graphAE_dataloader as Dataloader
from Models import conditional_model as conditional_model, graphAE as graphAE
from plyfile import PlyData
from utils import graphAE_param as Param

load_model_str = "../../train/0422_graphAE_dfaust/position/cond_model_0.pt"  # "../../train/0422_graphAE_dfaust/diffusion/trained_2000.pt"
output_dir = "../../train/0422_graphAE_dfaust/position/"
checkpoint_path =  output_dir + "cond_model_{}.pt"
in_sz = 3
n_steps = 100  # number of steps
num_steps = n_steps
batch_size = 32
betas = make_beta_schedule(schedule='sigmoid', n_timesteps=n_steps, start=1e-5, end=1e-2)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:0')
device = torch.device("cpu")

alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar = alphas_prod
alphas_bar_sqrt = torch.sqrt(alphas_prod).to(device)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod).to(device)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(device)

# Load dataset

def get_dloader():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Pad(int((256 - 224) / 2)),
    ])
    dataset = PositionsDataset(csv_file="../../data/positions/labels.csv", root_dir="../../data/positions/scene_silhouette_imgs", transform=preprocess)
    print("len(dataset):", len(dataset))
    dataset = [dataset[i] for i in range(0, 1000, 100)]
    dloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return dloader


def load_position_model(dsize):
    model = ConditionalModelPositionFromRGBScenesAndSilhouettes(
        n_steps, in_sz=in_sz, cond_sz_scene=64, cond_sz_silhouette=64, dsize=dsize, device=device)
    state_dict = torch.load("../../train/0422_graphAE_dfaust/position/cond_model_3999.pt")
    state_dict = parallel_to_cpu_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_shape_model():
    model = conditional_model.ConditionalModel(n_steps, in_sz=63, cond_sz=64, cond_model=True)
    state_dict = torch.load("../../train/0422_graphAE_dfaust/diffusion/cond_model_const_cond_4000.pt")
    state_dict = parallel_to_cpu_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    dloader = get_dloader()
    position_model = load_position_model(dsize=len(dloader.dataset))
    shape_model = load_shape_model()

    param_mesh = Param.Parameters()
    param_mesh.read_config("../../train/0422_graphAE_dfaust/30_conv_pool.config")
    param_mesh.batch = 1
    param_mesh.read_weight_path = "../../train/0422_graphAE_dfaust/weight_30/model_epoch0018.weight"
    mesh_model = graphAE.Model(param_mesh, test_mode=True)
    mesh_model.cuda()
    if param_mesh.read_weight_path != "":
        checkpoint = torch.load(param_mesh.read_weight_path)
        mesh_model.load_state_dict(checkpoint['model_state_dict'])
        mesh_model.init_test_mode()

    mesh_model.eval()
    template_plydata = PlyData.read(param_mesh.template_ply_fn)


    for batch in tqdm(dloader):
        
        # Generate position
        pos_sample = p_sample_loop(n_steps, position_model, [1, 3], alphas, one_minus_alphas_bar_sqrt, betas, batch['cond_position'])
        pos_sample = np.concatenate([s.detach().numpy() for s in pos_sample])
        
        # Generate shape
        shape_sample = p_sample_loop(n_steps, shape_model, [1, 63], alphas, one_minus_alphas_bar_sqrt, betas, batch['cond_shape'])
        shape_sample = np.concatenate([s.detach().numpy() for s in shape_sample])
        shape_sample = shape_sample.reshape(-1, 7, 9)
        shape_sample_torch = torch.FloatTensor(shape_sample).cuda()
        mesh = shape_sample_torch[-1]
        mesh = torch.unsqueeze(mesh, dim=0)
        out_mesh = mesh_model.forward_from_layer_n(mesh, 8)
        out_mesh = out_mesh.cpu()
        pc_out = np.array(out_mesh[0].data.tolist())
        pc_out -= pc_out.mean(axis=0)
        Dataloader.save_pc_into_ply(template_plydata, pc_out, "../../train/e2e/ball_gt_vs_pred/mesh.ply")
        
        # Generate screenshots
        
        x = batch['x']
        name = batch['name']
        trimeshes = scene_to_trimeshes(
            recording_name=batch['recording_name'][0],
            scene_name=batch['scene_name'][0],
            frame_id=batch['frame_id'][0])
        body = trimeshes['body']
        scene_mesh = trimeshes['scene']
        camera = trimeshes['camera']
        camera_pose = trimeshes['camera_pose']
        
        ## Save conditioning image
        img = PIL.Image.fromarray((batch['cond_shape'][0][0].numpy() * 255).astype(np.uint8), mode='L')
        # img = img.resize((int(1920 / 1), int(1080 / 1)))
        img.save(f'../../train/e2e/silhouettes/{name}.png')

        
        ## Generate silhouettes screenshot
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        scene.add(light, pose=camera_pose)
        scene.add(scene_mesh, 'scene_mesh')
        r = pyrender.OffscreenRenderer(viewport_width=1920,
                                        viewport_height=1080)
        color, depth = r.render(scene)
        color = color.astype(np.float32)
        silhouette = batch['cond_position']['silhouette'].squeeze()
        silhouette = torchvision.transforms.functional.resize(silhouette, (1080, 1920)).squeeze().permute((1,2,0))
        silhouette -= silhouette.min()
        silhouette /= silhouette.max()
        silhouette *= 255
        silhouette = silhouette.numpy()
        color = np.maximum(color, silhouette)
        img = PIL.Image.fromarray((color).astype(np.uint8))
        img = img.resize((int(1920 / 1), int(1080 / 1)))
        img.save(f'../../train/e2e/silhouettes_in_scene/{name}.png')
        
        
        ## Generate humans screenshot
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        scene.add(light, pose=camera_pose)
        body_gt = pyrender.Mesh.from_trimesh(body)
        scene.add(body_gt, 'body_gt')
        red_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=np.array([1.0, 0.0, 0.0, 1.0]))
        # Flip body
        body_pred = trimesh.load("../../train/e2e/ball_gt_vs_pred/mesh.ply")
        angle = math.pi
        direction = [1, 0, 0]
        center = [0, 0, 0]
        rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
        body_pred = body_pred.apply_transform(rot_matrix)
        body_pred = pyrender.Mesh.from_trimesh(body_pred, material=red_material)
        # matrix=np.eye(4)
        body_pred = pyrender.Node(mesh=body_pred, translation=pos_sample[-1])
        # body_pred.matrix[:3, 3] += pos_sample[-1]
        scene.add_node(body_pred)
        scene.add(scene_mesh, 'scene_mesh')
        r = pyrender.OffscreenRenderer(viewport_width=1920,
                                        viewport_height=1080)
        color, depth = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = PIL.Image.fromarray((color * 255).astype(np.uint8))
        img = img.resize((int(1920 / 1), int(1080 / 1)))
        img.save(f'../../train/e2e/body_gt_vs_pred/{name}.png')
        
        ## Generate scenes screenshot
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        scene.add(light, pose=camera_pose)
        scene.add(scene_mesh, 'scene_mesh')
        r = pyrender.OffscreenRenderer(viewport_width=1920,
                                        viewport_height=1080)
        color, depth = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = PIL.Image.fromarray((color * 255).astype(np.uint8))
        img = img.resize((int(1920 / 1), int(1080 / 1)))
        img.save(f'../../train/e2e/scenes/{name}.png')
        
        ## Generate predicted body screenshot
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        scene.add(light, pose=camera_pose)
        # Flip body
        body_pred = trimesh.load("../../train/e2e/ball_gt_vs_pred/mesh.ply")
        angle = math.pi
        direction = [1, 0, 0]
        center = [0, 0, 0]
        rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
        body_pred = body_pred.apply_transform(rot_matrix)
        body_pred = pyrender.Mesh.from_trimesh(body_pred)
        body_pred = pyrender.Node(mesh=body_pred, translation=pos_sample[-1])
        scene.add_node(body_pred)
        scene.add(scene_mesh, 'scene_mesh')
        r = pyrender.OffscreenRenderer(viewport_width=1920,
                                        viewport_height=1080)
        color, depth = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = PIL.Image.fromarray((color * 255).astype(np.uint8))
        img = img.resize((int(1920 / 1), int(1080 / 1)))
        img.save(f'../../train/e2e/body_pred/{name}.png')
        
        ## Generate groundtruth body screenshot
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        scene.add(light, pose=camera_pose)
        body_gt = pyrender.Mesh.from_trimesh(body)
        scene.add(body_gt, 'body_gt')
        scene.add(scene_mesh, 'scene_mesh')
        r = pyrender.OffscreenRenderer(viewport_width=1920,
                                        viewport_height=1080)
        color, depth = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = PIL.Image.fromarray((color * 255).astype(np.uint8))
        img = img.resize((int(1920 / 1), int(1080 / 1)))
        img.save(f'../../train/e2e/body_gt/{name}.png')
        
        ## Generate balls screenshot
        
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        scene.add(light, pose=camera_pose)
        body_ball_gt = pyrender.Mesh.from_trimesh(trimesh.primitives.Sphere(radius=0.1, center=x[0]))
        scene.add(body_ball_gt, 'body_ball_gt')
        red_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=np.array([1.0, 0.0, 0.0, 1.0]))
        body_ball_pred = pyrender.Mesh.from_trimesh(trimesh.primitives.Sphere(radius=0.1, center=pos_sample[-1]), material=red_material)
        scene.add(body_ball_pred, 'body_ball_pred')
        scene.add(scene_mesh, 'scene_mesh')
        r = pyrender.OffscreenRenderer(viewport_width=1920,
                                        viewport_height=1080)
        color, depth = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = PIL.Image.fromarray((color * 255).astype(np.uint8))
        img = img.resize((int(1920 / 1), int(1080 / 1)))
        img.save(f'../../train/e2e/ball_gt_vs_pred/{name}.png')

if __name__ == "__main__":
    main()