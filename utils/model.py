#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class UnFlatten(nn.Module):
    def forward(self, input, size):
        return input.view(input.size(0), size, 1, 1)
#%%
class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        
        self.config = config
        
        """encoder """
        encoder = []
        in_dim = 3
        for j in range(self.config["num_layer"]):
            encoder.append(nn.Conv2d(in_channels=in_dim, out_channels=self.config["hidden_dim"] * (1 + j), kernel_size=4, stride=2))
            encoder.append(nn.LeakyReLU(0.05))
            in_dim = self.config["hidden_dim"] * (1 + j)
        encoder.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder)

        self.feature_layer = nn.Linear(in_dim, self.config["latent_dim"])

        """weighted adjacency matrix"""
        p = {x:y for x,y in zip(range(config["latent_dim"]), range(config["latent_dim"]))}
        # build ReLU(Y)
        Y = torch.zeros((self.config["latent_dim"], self.config["latent_dim"]))
        for i in range(self.config["latent_dim"]):
            for j in range(self.config["latent_dim"]):
                Y[i, j] = p[j] - p[i]
        self.ReLU_Y = torch.nn.ReLU()(Y)

        self.W = nn.Parameter(self.ReLU_Y, requires_grad=True)

        """decoder"""
        decoder = []
        in_dim = self.config["latent_dim"]
        for j in reversed(range(1, self.config["num_layer"])):
            decoder.append(nn.ConvTranspose2d(in_dim, self.config["hidden_dim"] * (1 + j), kernel_size=4, stride=2))
            decoder.append(nn.LeakyReLU(0.05))
            in_dim = self.config["hidden_dim"] * (1 + j)
        decoder.append(nn.ConvTranspose2d(in_dim, 3, kernel_size=4, stride=2, padding=0))
        decoder.append(nn.Tanh())
        decoder.append(nn.ReflectionPad2d(1))
        self.decoder = nn.Sequential(*decoder)
    
    def forward(self, input):
        z = self.encoder(input)
        z = self.feature_layer(z)
        
        B_trans_z = torch.matmul(z, self.W * self.ReLU_Y)
        epsilon = torch.randn(B_trans_z.shape)
        z_sem = B_trans_z + epsilon
        xhat = self.decoder(UnFlatten()(z_sem, self.config["latent_dim"]))
        return z, B_trans_z, z_sem, xhat
#%%
def main():
    config = {
        "n": 100,
        "latent_dim": 4,
        "num_layer": 5,
        "hidden_dim": 8,
    }
    
    model = VAE(config)
    for x in model.parameters():
        print(x.shape)
        
    model.ReLU_Y
    
    batch = torch.rand(config["n"], 3, 96, 96)
    z, B_trans_z, z_sem, recon = model(batch)
    assert z.shape == (config["n"], config["latent_dim"])
    assert B_trans_z.shape == (config["n"], config["latent_dim"])
    assert z_sem.shape == (config["n"], config["latent_dim"])
    assert recon.shape == (config["n"], 3, 96, 96)
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# from PIL import Image
# img = Image.open("/Users/anseunghwan/Documents/GitHub/causal_vae/utils/causal_data/pendulum/train/a_-1_60_6_5.png")
# np.array(img)[:, :, :3]
#%%
# x = torch.rand(4, 3, 96, 96)

# # encoder 
# encoder = []
# in_dim = 3
# for j in range(config["num_layer"]):
#     encoder.append(nn.Conv2d(in_channels=in_dim, out_channels=config["hidden_dim"] * (1 + j), kernel_size=4, stride=2))
#     encoder.append(nn.LeakyReLU(0.05))
#     in_dim = config["hidden_dim"] * (1 + j)
# encoder.append(nn.Flatten())
# encoder = nn.Sequential(*encoder)

# h = encoder(x)
# h.shape

# feature_layer = nn.Linear(40, config["latent_dim"])
# h = feature_layer(h)

# # weighted adjacency matrix
# p = {x:y for x,y in zip(np.arange(config["latent_dim"]), np.arange(config["latent_dim"]))}
# """build ReLU(Y)"""
# Y = torch.zeros((config["latent_dim"], config["latent_dim"]))
# for i in range(config["latent_dim"]):
#     for j in range(config["latent_dim"]):
#         Y[i, j] = p[j] - p[i]
# ReLU_Y = torch.nn.ReLU()(Y)

# W = torch.rand(config["latent_dim"], config["latent_dim"])
# min = -0.1
# max = 0.1
# W = (max - min) * W + min # ~ Uniform(-0.1, 0.1)
# W = W.fill_diagonal_(0.)
# W = nn.Parameter(W, requires_grad=True)

# B_trans_h = torch.matmul(h, W * ReLU_Y)

# class UnFlatten(nn.Module):
#     def forward(self, input, size=config["latent_dim"]):
#         return input.view(input.size(0), size, 1, 1)

# epsilon = torch.randn(B_trans_h.shape)
# h = UnFlatten()(B_trans_h + epsilon)

# # decoder
# decoder = []
# in_dim = config["latent_dim"]
# for j in reversed(range(1, config["num_layer"])):
#     decoder.append(nn.ConvTranspose2d(in_dim, config["hidden_dim"] * (1 + j), kernel_size=4, stride=2))
#     decoder.append(nn.LeakyReLU(0.05))
#     in_dim = config["hidden_dim"] * (1 + j)
# # decoder.append(nn.Flatten())
# decoder.append(nn.ConvTranspose2d(in_dim, 3, kernel_size=4, stride=2, padding=0))
# decoder.append(nn.ReflectionPad2d(1))
# decoder = nn.Sequential(*decoder)
# decoder(h).shape
#%%