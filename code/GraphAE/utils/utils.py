import torch
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from sklearn.datasets import make_swiss_roll
import torch
import numpy as np
from Models import graphAE as graphAE
import trimesh
import pyrender
from PIL import Image
from os import listdir
from os.path import isfile, join
import natsort
from tqdm import tqdm


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



def sample_batch(size, noise=0.5):
    x, _ = make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0


def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-1, end=1e-1):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas


def extract(input, t, shape):
    out = torch.gather(input, dim=0, index=t.to(input.device))  # get value at specified t
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    color = mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
    (r, g, b) = mpl.colors.ColorConverter.to_rgb(color)
    return np.array([r, g, b])


def get_colors_from_diff_pc(diff_pc, min_error, max_error):
    # colors = np.zeros((diff_pc.shape[0], 3))
    mix = (diff_pc - min_error) / (max_error - min_error)
    mix = np.clip(mix, 0, 1)  # point_num
    cmap = cm.get_cmap('coolwarm')
    colors = cmap(mix)[:, 0:3]
    return colors


def get_faces_colors_from_vertices_colors(vertices_colors, faces):
    faces_colors = vertices_colors[faces]
    faces_colors = faces_colors.mean(1)
    return faces_colors


def get_faces_from_ply(ply):
    faces_raw = ply['face']['vertex_indices']
    faces = np.zeros((faces_raw.shape[0], 3)).astype(np.int32)
    for i in range(faces_raw.shape[0]):
        faces[i][0] = faces_raw[i][0]
        faces[i][1] = faces_raw[i][1]
        faces[i][2] = faces_raw[i][2]

    return faces


def p_sample_loop(n_steps, model, shape, alphas, one_minus_alphas_bar_sqrt, betas, cond):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, alphas, one_minus_alphas_bar_sqrt, betas, cond)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, alphas, one_minus_alphas_bar_sqrt, betas, cond):
    t = torch.tensor([t])
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x.shape)) / extract(one_minus_alphas_bar_sqrt, t, x.shape))
    # Model output
    eps_theta = model(x, t, cond=cond)
    # Final values
    mean = (1 / extract(alphas, t, x.shape).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x)
    # Fixed sigma
    sigma_t = extract(betas, t, x.shape).sqrt()
    sample = mean + sigma_t * z
    return (sample)

def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))

def noise_estimation_loss(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, device, cond):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long().to(device)
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, x_0.shape)
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, x_0.shape)
    e = torch.randn_like(x_0)
    # model input
    x = x_0 * a + e * am1
    output = model(x, t, cond=cond)
    return (e - output).square().mean()

def parallel_to_cpu_state_dict(state_dict):
    deparalleled_state_dict = {}
    for k in state_dict.keys():
        s = "cond_model.module"
        new_k = k
        if k.startswith(s):
            new_k = "cond_model" + k[len(s):]
        deparalleled_state_dict[new_k] = state_dict[k]
    return deparalleled_state_dict


def img_folder_to_np(path, create_silhouettes=True):
    img_filenames = [f for f in listdir(path) if isfile(join(path, f))]
    img_filenames = natsort.natsorted(img_filenames,reverse=False)
    arrays = []
    for img_filename in tqdm(img_filenames):
        with Image.open(path + img_filename) as img:
            img = img.resize((224, 224))
            arr = np.array(img)
            if len(arr.shape) == 3:
                arr = arr[:,:,0] == 255
            else:
                arr = arr == 255
            arrays.append(arr)
    arr = np.stack(arrays, axis=0)
    return arr