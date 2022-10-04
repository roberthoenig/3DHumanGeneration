import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from Models import EMA as EMA
from Models.conditional_model import ConditionalModel
from utils import make_beta_schedule, noise_estimation_loss

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ConditionalModel(n_steps)
model.load_state_dict(torch.load("model.pt"))
model.eval()
# model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Create EMA model
ema = EMA(0.9)
ema.register(model)
# Batch size
batch_size = 16

# if __name__ == 'main':
data = np.load("../../data/DFAUST/test_res.npy")
indices = list(range(0, 960, 32))
data = data[indices, :, :, :]
data = np.reshape(data, (960, 7, 9))
data = data.reshape((960, 63))
dataset = torch.Tensor(data).float()
print(dataset.shape)

# dataset = dataset.to(device)


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
        batch_x = dataset[indices]
        # Compute the loss.
        loss = noise_estimation_loss(model, batch_x)
        # Before the backward pass, zero all of the network gradients
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        # Calling the step function to update the parameters
        optimizer.step()
        # Update the exponential moving average
        ema.update(model)
        losses.append(loss.detach().item())
    batch_loss = np.array(losses).mean()
    pbar.set_postfix({'batch_loss': batch_loss})
    batch_losses.append(batch_loss)

plt.plot(batch_losses)
plt.savefig("batch_losses")
torch.save(model.state_dict(), "../../train/0422_graphAE_dfaust/diffusion/model.pt")
