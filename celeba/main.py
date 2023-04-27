#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt

from datasets import CelebALoader
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
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
    parser.add_argument('--batch_size', default=16, type=int,
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
config["cuda"] = torch.cuda.is_available()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#%%
dataset = CelebALoader(config) # smile
random_sampler = RandomSampler(dataset)
batch_sampler = BatchSampler(random_sampler, batch_size=config["batch_size"], drop_last=True)
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

"""mask"""
mask = torch.split(batch[0][..., 3:], 1, dim=-1)
#%%
import importlib
model_module = importlib.import_module('module.model')
importlib.reload(model_module)

model = getattr(model_module, config["model"])(B, mask, config, device).to(device)
model.train()
#%%
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config["lr"]
)

from module.train import train_CDGVAE

for epoch in range(config["epochs"]):
    logs, xhat = train_CDGVAE(train_loader, model, config, optimizer, device)
    
    print_input = "[epoch {:03d}]".format(epoch + 1)
    print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
    print(print_input)
    
    if epoch % 10 == 0:
        plt.figure(figsize=(4, 4))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
            plt.axis('off')
        plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
        plt.close()
#%%
# for param in model.encoder.parameters():
#     print(param.requires_grad)
#%%
# x_batch = x_batch.cuda()
# y_batch = y_batch.cuda()

# (mean1, logvar1, epsilon1, orig_latent, latent, logdet), (mean2, logvar2, epsilon2), align_latent, xhat_separated, xhat = model(x_batch)
# plt.imshow((xhat[0].cpu().detach() + 1) / 2)
#%%