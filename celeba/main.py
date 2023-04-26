#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt

from datasets import CelebALoader
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from module.model import CDGVAE
from module.sagan import *
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--model', type=str, default='CDGVAE', 
                        help='VAE based model options: VAE, InfoMax, CDGVAE')
    parser.add_argument('--causal_structure', type=float, default=0, 
                        help='0 or 1')

    # causal structure
    parser.add_argument("--node", default=6, type=int,
                        help="the number of nodes")
    parser.add_argument("--latent_dim", default=6, type=int,
                        help="latent dimension size")
    parser.add_argument("--scm", default='linear', type=str,
                        help="SCM structure options: linear or nonlinear")
    parser.add_argument("--flow_num", default=1, type=int,
                        help="the number of invertible NN flow")
    parser.add_argument("--inverse_loop", default=100, type=int,
                        help="the number of inverse loop")
    
    # data options
    parser.add_argument('--labeled_ratio', default=1, type=float, # fully-supervised
                        help='ratio of labeled dataset for semi-supervised learning')
    
    parser.add_argument("--label_normalization", default=True, type=bool,
                        help="If True, normalize additional information label data")
    parser.add_argument("--adjacency_scaling", default=True, type=bool,
                        help="If True, scaling adjacency matrix with in-degree")
    parser.add_argument('--img_size', default=128, type=int,
                        help='width and heigh of image')
    
    # optimization options
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    # loss coefficients
    parser.add_argument('--beta', default=0.1, type=float,
                        help='observation noise')
    parser.add_argument('--lambda', default=5, type=float,
                        help='weight of label alignment loss')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
config = vars(get_args(debug=True)) # default configuration
#%%
dataset = CelebALoader(config) # smile
random_sampler = RandomSampler(dataset)
batch_sampler = BatchSampler(random_sampler, batch_size=config["batch_size"], drop_last=False)
train_loader = DataLoader(
    dataset, shuffle=False, pin_memory=True, batch_sampler=batch_sampler)
#%%
for batch in train_loader:
    break
#%%
batch[0].shape
batch[1].shape
#%%
i = 0
fig, ax = plt.subplots(1, 6, figsize=(20, 5))
ax[0].imshow(batch[0][i][..., :3])
for j in range(5):
    ax[1+j].imshow(batch[0][i][..., 3+j])
#%%
B = torch.zeros(config["node"], config["node"])

if config["causal_structure"] == 0:
    B[dataset.nodes.index('Smiling'), dataset.nodes.index('High_Cheekbones')] = 1
    B[dataset.nodes.index('Smiling'), dataset.nodes.index('Mouth_Slightly_Open')] = 1
    B[dataset.nodes.index('Smiling'), dataset.nodes.index('Chubby')] = 1
    B[dataset.nodes.index('Smiling'), dataset.nodes.index('Narrow_Eyes')] = 1
    B[dataset.nodes.index('Male'), dataset.nodes.index('Narrow_Eyes')] = 1
elif config["causal_structure"] == 1:
    B[dataset.nodes.index('Young'), dataset.nodes.index('Bags_Under_Eyes')] = 1
    B[dataset.nodes.index('Young'), dataset.nodes.index('Chubby')] = 1
    B[dataset.nodes.index('Young'), dataset.nodes.index('Heavy_Makeup')] = 1
    B[dataset.nodes.index('Young'), dataset.nodes.index('Receding_Hairline')] = 1
    B[dataset.nodes.index('Male'), dataset.nodes.index('Heavy_Makeup')] = 1
    B[dataset.nodes.index('Male'), dataset.nodes.index('Receding_Hairline')] = 1
else:
    raise ValueError('Not supported causal structure!')

"""adjacency matrix scaling"""
if config["adjacency_scaling"]:
    indegree = B.sum(axis=0)
    mask = (indegree != 0)
    B[:, mask] = B[:, mask] / indegree[mask]
#%%
from module.resnet import *
encoder = resnet50(
    pretrained=False, in_channels=3, fc_size=2048, 
    out_dim=config["node"] * 2 + config["latent_dim"] * 2)

class Generator(nn.Module):
    r'''SAGAN Generator

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        out_channels: number of output channels
        add_noise: whether to add noises to each conv layer
        attn: whether to add self-attention layer
    '''

    def __init__(self, latent_dim, conv_dim=32, image_size=128, out_channels=3, add_noise=True, attn=True):
        super().__init__()

        self.latent_dim = latent_dim
        self.conv_dim = conv_dim
        self.image_size = image_size
        self.add_noise = add_noise
        self.attn = attn

        self.block0 = GenIniBlock(latent_dim, conv_dim * 16, 4, add_noise=add_noise)
        self.block1 = GenBlock(conv_dim * 16, conv_dim * 16, size=8, add_noise=add_noise)
        self.block2 = GenBlock(conv_dim * 16, conv_dim * 8, size=16, add_noise=add_noise)
        if image_size == 64:
            self.block3 = GenBlock(conv_dim * 8, conv_dim * 4, size=32, add_noise=add_noise)
            if attn:
                self.self_attn1 = Self_Attn(conv_dim * 4)
            self.block4 = GenBlock(conv_dim * 4, conv_dim * 2, size=64, add_noise=add_noise)
            conv_dim = conv_dim * 2
        elif image_size == 128:
            self.block3 = GenBlock(conv_dim * 8, conv_dim * 4, add_noise=add_noise)
            if attn:
                self.self_attn1 = Self_Attn(conv_dim * 4)
            self.block4 = GenBlock(conv_dim * 4, conv_dim * 2, add_noise=add_noise)
            # self.self_attn2 = Self_Attn(conv_dim*2)
            self.block5 = GenBlock(conv_dim * 2, conv_dim, add_noise=add_noise)
        else: # image_size == 256 or 512
            self.block3 = GenBlock(conv_dim * 8, conv_dim * 8, add_noise=add_noise)
            self.block4 = GenBlock(conv_dim * 8, conv_dim * 4, add_noise=add_noise)
            if attn:
                self.self_attn1 = Self_Attn(conv_dim * 4)
            self.block5 = GenBlock(conv_dim * 4, conv_dim * 2, add_noise=add_noise)
            self.block6 = GenBlock(conv_dim * 2, conv_dim, add_noise=add_noise)
            if image_size == 512:
                self.block7 = GenBlock(conv_dim, conv_dim, add_noise=add_noise)

        self.bn = nn.BatchNorm2d(conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.toRGB = snconv2d(in_channels=conv_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z):
        out = self.block0(z)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        if self.attn:
            out = self.self_attn1(out)
        out = self.block4(out)
        if self.image_size > 64:
            out = self.block5(out)
            if self.image_size == 256 or self.image_size == 512:
                out = self.block6(out)
                if self.image_size == 512:
                    out = self.block7(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.toRGB(out)
        out = self.tanh(out)
        return out

decoder = [
    Generator(2), Generator(2), Generator(2), Generator(3), Generator(config["latent_dim"])]
#%%
h, _ = encoder(batch[0][..., :3].permute(0, 3, 1, 2))
h1, h2 = torch.split(h, [config["node"] * 2, config["latent_dim"] * 2], dim=1)
mean1, logvar1 = torch.split(h1, config["node"], dim=1)
mean2, logvar2 = torch.split(h2, config["latent_dim"], dim=1)
#%%
"""Causal Adjacency Matrix"""
device = 'cpu'
I = torch.eye(config["node"])
I_B_inv = torch.inverse(I - B)

"""Generalized Linear SEM: Invertible NN"""
if config["scm"] == "linear":
    flows = nn.ModuleList(
        [InvertiblePriorLinear(device=device) for _ in range(config["node"])])
elif config["scm"] == "nonlinear":
    flows = nn.ModuleList(
        [PlanarFlows(1, config["flow_num"], config["inverse_loop"], device) for _ in range(config["node"])])
else:
    raise ValueError('Not supported SCM!')
#%%
noise = torch.randn(batch[0].size(0), config["node"])
epsilon1 = mean1 + torch.exp(logvar1 / 2) * noise
epsilon2 = mean2 + torch.exp(logvar2 / 2) * noise
#%%
latent = torch.matmul(epsilon1, I_B_inv) # [batch, node], input = epsilon (exogenous variables)
orig_latent = latent.clone()
latent = torch.split(latent, 1, dim=1) # [batch, 1] x node
latent = list(map(lambda x, layer: layer(x, log_determinant=False), latent, flows)) # input = (I-B^T)^{-1} * epsilon
logdet = [x[1] for x in latent]
latent = [x[0] for x in latent]
#%%
latent = [
    torch.cat([latent[0], latent[2]], dim=1),
    torch.cat([latent[0], latent[3]], dim=1),
    torch.cat([latent[0], latent[4]], dim=1),
    torch.cat([latent[0], latent[1], latent[5]], dim=1),
    epsilon2]
#%%
xhat_separated = [D(z) for D, z in zip(decoder, latent)]
xhat = [x.permute(0, 2, 3, 1) for x in xhat_separated]
#%%
mask = torch.split(batch[0][..., 3:], 1, dim=-1)[0]
xhat_ = [x for x, m in zip(xhat, mask)] # masking
#%%
xhat_ = torch.tanh(sum(xhat_))
plt.imshow((xhat_[0].detach() + 1) / 2)
#%%