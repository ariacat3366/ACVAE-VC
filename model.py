import torch
import torch.nn as nn
import torch.nn.functional as F
        
class ACVAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1+4, 8, (3,9), (1,1))
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8+4, 16, (4,8), (2,2))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(12+4, 16, (4,8), (2,2))
        self.conv3_bn = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16+4, 10, (9,5), (9,1))
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(5+4, 16, (9,5), (9,1))
        self.upconv1_bn = nn.BatchNorm2d(16)
        self.upconv2 = nn.ConvTranspose2d(16+4, 16, (4,8), (2,2))
        self.upconv2_bn = nn.BatchNorm2d(16)
        self.upconv3 = nn.ConvTranspose2d(16+4, 8, (4,8), (2,2))
        self.upconv3_bn = nn.BatchNorm2d(8)
        self.upconv4 = nn.ConvTranspose2d(8+4, 2, (9,5), (1,1))
        
        # Auxiliary Classifier
        self.ac_conv1 = nn.Conv2d(1, 8, (4,4), (2,2))
        self.ac_conv1_bn = nn.BatchNorm2d(8)
        self.ac_conv2 = nn.Conv2d(8, 16, (4,4), (2,2))
        self.ac_conv2_bn = nn.BatchNorm2d(16)
        self.ac_conv3 = nn.Conv2d(16, 32, (4,4), (2,2))
        self.ac_conv3_bn = nn.BatchNorm2d(32)
        self.ac_conv4 = nn.Conv2d(32, 16, (4,4), (2,2))
        self.ac_conv4_bn = nn.BatchNorm2d(16)
        self.ac_conv5 = nn.Conv2d(16, 4, (1,4), (1,2))

    def encode(self, x, label):
        
        h1 = F.glu(self.conv1_bn(self.conv1(torch.cat((x, label), dim=1))))
        
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def one_hot_label(self, label, shape):
        label_layer = np.zeros(shape)
        label_layer[:,label] = np.ones((shape[-1:]))
        return label_layer
        
    
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD








