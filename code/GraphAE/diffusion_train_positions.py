import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torch.optim as optim
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from Models.EMA import EMA as EMA
from Models.conditional_model import ConditionalModel, ConditionalModelPositionFromRGBScenesAndSilhouettes
from utils.utils import make_beta_schedule, noise_estimation_loss, parallel_to_cpu_state_dict
from positions_dataset import PositionsDataset

type = 'position'  # (out of 'position', 'shape', 'none')
load_model_str = "../../train/0422_graphAE_dfaust/position/cond_model_0.pt"  # "../../train/0422_graphAE_dfaust/diffusion/trained_2000.pt"
output_dir = "../../train/0422_graphAE_dfaust/position/"
checkpoint_path =  output_dir + "cond_model_{}.pt"
in_sz = 3
n_steps = 100  # number of steps
num_steps = n_steps
batch_size = 32
# betas = make_beta_schedule(schedule='linear', n_timesteps=num_steps, start=1e-3, end=1e-3)
# betas = make_beta_schedule(schedule='linear', n_timesteps=num_steps, start=1e-3, end=1e-3)
betas = make_beta_schedule(schedule='sigmoid', n_timesteps=n_steps, start=1e-5, end=1e-2)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:0')
device = torch.device("cpu")

# betas = make_beta_schedule(schedule='sigmoid', n_timesteps=num_steps, start=1e-5, end=1e-2)
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar = alphas_prod
alphas_bar_sqrt = torch.sqrt(alphas_prod).to(device)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod).to(device)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(device)

print(one_minus_alphas_bar_log)

if type == 'shape' or type == 'none':
    data = np.load("../../data/DFAUST/test_res.npy")
    # dataloader = TODO
elif type == 'position':
    # data = np.load("../../data/DFAUST/positions.npy")
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
    dloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
else:
    raise Exception(f"Unkown type {type}")

# indices = list(range(0, 32928, 32))
# data = data[indices, :, :, :]
# data = np.reshape(data, (32928, 7, 9))
# data = data.reshape((32928, 63))
# dataset = torch.Tensor(data).float()
# print(dataset.shape)

# dataset = dataset.to(device)

# handle conditioning
if type == 'shape':
    cond_dataset = np.load("../../data/DFAUST/test_res_silhouettes.npy")
    cond_dataset = torch.Tensor(cond_dataset).float()
    cond_dataset = cond_dataset.unsqueeze(1)
    model = ConditionalModel(n_steps, in_sz=in_sz, cond_sz=64, cond_model=True)
elif type == 'position':
    pass
    # cond_dataset = np.load("../../data/DFAUST/rgb_scenes_silhouettes.npy")
    # cond_dataset = torch.Tensor(cond_dataset).float()
    # cond_dataset = cond_dataset.unsqueeze(1)
    model = ConditionalModelPositionFromRGBScenesAndSilhouettes(
        n_steps, in_sz=in_sz, cond_sz_scene=64, cond_sz_silhouette=64, dsize=len(dataset), device=device)
elif type == 'none':
    cond_dataset = torch.zeros(*data.shape[:-1], 0)
    model = ConditionalModel(n_steps, in_sz=in_sz, cond_sz=0, cond_model=False)
else:
    raise Exception(f"Unkown type {type}.")

if load_model_str is not None:
    model_state_dict = torch.load(load_model_str)
    model.load_state_dict(parallel_to_cpu_state_dict(model_state_dict))

# model.load_state_dict(torch.load("../../train/0422_graphAE_dfaust/diffusion/model.pt"))
model.eval()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Create EMA model  
ema = EMA(0.9)
ema.register(model)


batch_losses = []

n_steps_training = 5_000
pbar = tqdm(range(n_steps_training))
for t in pbar:
    # X is a torch Variable
    # permutation = torch.randperm(dataset.size()[0])
    losses = []
    for batch in dloader:
        # Retrieve current batch
        # Compute the loss.
        loss = noise_estimation_loss(model, batch['x'], alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, device, cond=batch['cond'], idx=batch['idx'])
        # Before the backward pass, zero all of the network gradients
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        # Calling the step function to update the parameters
        optimizer.step()
        # Update the exponential moving average
        # ema.update(model)
        losses.append(loss.detach().item())
    batch_loss = np.array(losses).mean()
    pbar.set_postfix({'batch_loss': batch_loss})
    batch_losses.append(batch_loss)
    if (t+1) % 1000 == 0 or t == 1:
        print(f"saving to train/0422_graphAE_dfaust/diffusion/cond_model_{t}.pt")
        torch.save(model.cpu().state_dict(), checkpoint_path.format(t)) 
        model.to(device)

plt.yscale('log')
plt.plot(batch_losses)
plt.yscale('log')
plt.savefig(output_dir+"batch_losses_cond.png")
torch.save(model.state_dict(), output_dir + "cond_model_fully_trained.pt")
