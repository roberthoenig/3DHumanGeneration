import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, n_steps, in_sz, cond_sz, cond_model):
        super(ConditionalModel, self).__init__()
        if cond_model == True:
            cond_model = torchvision.models.squeezenet1_1(pretrained=True)
            cond_model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2))
            cond_model.classifier = torch.nn.Sequential(
                torch.nn.AvgPool2d(13, 13),
                torch.nn.Flatten(),
                torch.nn.Linear(512, cond_sz),
            )
            cond_model = torch.nn.Sequential(
                torchvision.transforms.Resize(224),
                torchvision.transforms.Pad(int((256 - 224) / 2)),
                cond_model,
            )
        else:
            cond_model = None
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            cond_model = torch.nn.DataParallel(cond_model)
        self.cond_model = cond_model
        for param in self.cond_model.parameters():
            param.requires_grad = False
        self.lin1 = ConditionalLinear(in_sz+cond_sz, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, in_sz)

    def forward(self, x, y, cond):
        if self.cond_model is not None:
            cond_processed = self.cond_model(cond)
            x = torch.cat([x, cond_processed], dim=-1)
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

def print_x(x):
    print("lambda x:", x.shape)
    return x


class ConditionalModelPositionFromRGBScenesAndSilhouettes(nn.Module):
    def __init__(self, n_steps, in_sz, cond_sz_scene, cond_sz_silhouette, dsize, device):
        super(ConditionalModelPositionFromRGBScenesAndSilhouettes, self).__init__()
        self.img_model_scene = torchvision.models.squeezenet1_1(pretrained=True) 
        self.device = device
        self.img_model_scene.classifier = torch.nn.Sequential(
            torch.nn.AvgPool2d(15, 15),
            torch.nn.Flatten(),
            torch.nn.Linear(512, cond_sz_scene),
        )
        self.img_model_silhouette = torchvision.models.squeezenet1_1(pretrained=True)
        self.img_model_silhouette.classifier = torch.nn.Sequential(
            torch.nn.AvgPool2d(15, 15),
            torch.nn.Flatten(),
            torch.nn.Linear(512, cond_sz_silhouette),
        )
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            img_model_silhouette = torch.nn.DataParallel(img_model_silhouette)
            img_model_scene = torch.nn.DataParallel(img_model_scene)
        for param in self.img_model_silhouette.parameters():
            param.requires_grad = False
        for param in self.img_model_scene.parameters():
            param.requires_grad = False
        self.lin1 = ConditionalLinear(in_sz+cond_sz_scene+cond_sz_silhouette, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, in_sz)
        self.cache = torch.zeros(dsize, cond_sz_scene+cond_sz_silhouette)
        self.idx_stores = set()

    def forward(self, x, y, cond, idx=None):
        if idx is not None:
            all_present = True
            for i in idx:
                if not i.item() in self.idx_stores:
                    all_present = False
                    break
            if all_present == False:
                cond_img_scene = self.img_model_scene(cond['scene'].to(self.device))
                cond_img_silhouette = self.img_model_silhouette(cond['silhouette'].to(self.device))
                self.cache[idx,:] = torch.cat([cond_img_scene, cond_img_silhouette], dim=-1)
                for i in idx:
                    self.idx_stores.add(i.item())
            cond = self.cache[idx]
        else:
            cond_img_scene = self.img_model_scene(cond['scene'].to(self.device))
            cond_img_silhouette = self.img_model_silhouette(cond['silhouette'].to(self.device))
            cond = torch.cat([cond_img_scene, cond_img_silhouette], dim=-1)

        x = torch.cat([x, cond], dim=-1)
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)
