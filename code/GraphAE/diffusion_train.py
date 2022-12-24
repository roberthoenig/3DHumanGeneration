import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.optim as optim
import itertools
from tqdm import tqdm

from Models.EMA import EMA as EMA
from Models.conditional_model import ConditionalModel
from utils.utils import make_beta_schedule, noise_estimation_loss, parallel_to_cpu_state_dict

condition_on_images = True
in_sz = 63
n_steps = 100  # number of steps
num_steps = n_steps
# betas = make_beta_schedule(schedule='linear', n_timesteps=num_steps, start=1e-3, end=1e-3)
# betas = make_beta_schedule(schedule='linear', n_timesteps=num_steps, start=1e-3, end=1e-3)
betas = make_beta_schedule(schedule='sigmoid', n_timesteps=n_steps, start=1e-5, end=1e-2)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
# device = torch.device("cpu")

# betas = make_beta_schedule(schedule='sigmoid', n_timesteps=num_steps, start=1e-5, end=1e-2)
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar = alphas_prod
alphas_bar_sqrt = torch.sqrt(alphas_prod).to(device)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod).to(device)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(device)

# create dataset
# Batch size
batch_size = 64

# if __name__ == 'main':
data = np.load("../../data/DFAUST/test_res.npy")
print("data.shape", data.shape)

indices = list(range(0, 32928, 32))
data = data[indices, :, :, :]
data = np.reshape(data, (32928, 7, 9))
data = data.reshape((32928, 63))
dataset = torch.Tensor(data).float()
print(dataset.shape)

# dataset = dataset.to(device)

# handle conditioning
if condition_on_images:
    cond_dataset = np.load("../../data/DFAUST/test_res_silhouettes.npy")
    print("cond_dataset.shape", cond_dataset.shape)
    cond_dataset = torch.Tensor(cond_dataset).float()
    cond_dataset = cond_dataset.unsqueeze(1)
    print("cond_dataset.shape", cond_dataset.shape)
    cond_sz = 64
else:
    cond_dataset = torch.zeros(*data.shape[:-1], 0)
    cond_sz = 0

model = ConditionalModel(n_steps, in_sz=in_sz, cond_sz=cond_sz, cond_model=condition_on_images)
load_model = False
if load_model:
    model_state_dict = torch.load("../../train/0422_graphAE_dfaust/diffusion/trained_2000.pt")
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
    permutation = torch.randperm(dataset.size()[0])
    losses = []
    for i in range(0, dataset.size()[0], batch_size):
        # Retrieve current batch
        indices = permutation[i:i + batch_size]
        batch_x = dataset[indices].to(device)
        cond_x = cond_dataset[indices].float().to(device)
        # Compute the loss.
        loss = noise_estimation_loss(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, device, cond=cond_x)
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
    if (t+1) % 1000 == 0:
        print(f"saving to train/0422_graphAE_dfaust/diffusion/cond_model_{t}.pt")
        torch.save(model.cpu().state_dict(), f"../../train/0422_graphAE_dfaust/diffusion/cond_model_{t}.pt") 
        model.to(device)

plt.yscale('log')
plt.plot(batch_losses)
plt.yscale('log')
plt.savefig("../../train/0422_graphAE_dfaust/diffusion/batch_losses_cond")
torch.save(model.state_dict(), "../../train/0422_graphAE_dfaust/diffusion/cond_model.pt")
