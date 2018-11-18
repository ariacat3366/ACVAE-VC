import torch
import torch.nn as nn
import torch.nn.functional as F
        
class ACVAE(nn.Module):
    def __init__(self):
        
        self.label_num = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super(ACVAE, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1+self.label_num, 8, (3,9), (1,1), padding=(1, 4))
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_gated = nn.Conv2d(1+self.label_num, 8, (3,9), (1,1), padding=(1, 4))
        self.conv1_gated_bn = nn.BatchNorm2d(8)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(8+self.label_num, 16, (4,8), (2,2), padding=(1, 3))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2_gated = nn.Conv2d(8+self.label_num, 16, (4,8), (2,2), padding=(1, 3))
        self.conv2_gated_bn = nn.BatchNorm2d(16)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(16+self.label_num, 16, (4,8), (2,2), padding=(1, 3))
        self.conv3_bn = nn.BatchNorm2d(16)
        self.conv3_gated = nn.Conv2d(16+self.label_num, 16, (4,8), (2,2), padding=(1, 3))
        self.conv3_gated_bn = nn.BatchNorm2d(16)
        self.conv3_sigmoid = nn.Sigmoid()
        
        self.conv4_mu = nn.Conv2d(16+self.label_num, 10//2, (9,5), (9,1), padding=(1, 2))
        self.conv4_logvar = nn.Conv2d(16+self.label_num, 10//2, (9,5), (9,1), padding=(1, 2))
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(5+self.label_num, 16, (9,5), (9,1))
        self.upconv1_bn = nn.BatchNorm2d(16)
        self.upconv1_gated = nn.ConvTranspose2d(5+self.label_num, 16, (9,5), (9,1))
        self.upconv1_gated_bn = nn.BatchNorm2d(16)
        self.upconv1_sigmoid = nn.Sigmoid()
        
        self.upconv2 = nn.ConvTranspose2d(16+self.label_num, 16, (4,8), (2,2))
        self.upconv2_bn = nn.BatchNorm2d(16)
        self.upconv2_gated = nn.ConvTranspose2d(16+self.label_num, 16, (4,8), (2,2))
        self.upconv2_gated_bn = nn.BatchNorm2d(16)
        self.upconv2_sigmoid = nn.Sigmoid()
        
        self.upconv3 = nn.ConvTranspose2d(16+self.label_num, 8, (4,8), (2,2))
        self.upconv3_bn = nn.BatchNorm2d(8)
        self.upconv3_gated = nn.ConvTranspose2d(16+self.label_num, 8, (4,8), (2,2))
        self.upconv3_gated_bn = nn.BatchNorm2d(8)
        self.upconv3_sigmoid = nn.Sigmoid()
        
        self.upconv4_mu = nn.ConvTranspose2d(8+self.label_num, 2//2, (9,5), (1,1))
        self.upconv4_logvar = nn.ConvTranspose2d(8+self.label_num, 2//2, (9,5), (1,1))
        
        # Auxiliary Classifier
        self.ac_conv1 = nn.Conv2d(1, 8, (4,4), (2,2))
        self.ac_conv1_bn = nn.BatchNorm2d(8)
        self.ac_conv1_gated = nn.Conv2d(1, 8, (4,4), (2,2))
        self.ac_conv1_gated_bn = nn.BatchNorm2d(8)
        self.ac_conv1_sigmoid = nn.Sigmoid()
        
        self.ac_conv2 = nn.Conv2d(8, 16, (4,4), (2,2))
        self.ac_conv2_bn = nn.BatchNorm2d(16)
        self.ac_conv2_gated = nn.Conv2d(8, 16, (4,4), (2,2))
        self.ac_conv2_gated_bn = nn.BatchNorm2d(16)
        self.ac_conv2_sigmoid = nn.Sigmoid()
        
        self.ac_conv3 = nn.Conv2d(16, 32, (4,4), (2,2))
        self.ac_conv3_bn = nn.BatchNorm2d(32)
        self.ac_conv3_gated = nn.Conv2d(16, 32, (4,4), (2,2))
        self.ac_conv3_gated_bn = nn.BatchNorm2d(32)
        self.ac_conv3_sigmoid = nn.Sigmoid()
        
        self.ac_conv4 = nn.Conv2d(32, 16, (4,4), (2,2))
        self.ac_conv4_bn = nn.BatchNorm2d(16)
        self.ac_conv4_gated = nn.Conv2d(32, 16, (4,4), (2,2))
        self.ac_conv4_gated_bn = nn.BatchNorm2d(16)
        self.ac_conv4_sigmoid = nn.Sigmoid()
        
        self.ac_conv5 = nn.Conv2d(16, self.label_num, (1,4), (1,2))
        self.ac_fc5 = nn.Linear(self.label_num * 32, self.label_num)

    def encode(self, x, label):
       
        h1_ = self.conv1_bn(self.conv1(self.concat_label(x, label)))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(self.concat_label(x, label)))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 
        
        h2_ = self.conv2_bn(self.conv2(self.concat_label(h1, label)))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(self.concat_label(h1, label)))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
        
        h3_ = self.conv3_bn(self.conv3(self.concat_label(h2, label)))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(self.concat_label(h2, label)))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 
        
        h4_mu = self.conv4_mu(self.concat_label(h3, label))
        h4_logvar = self.conv4_logvar(self.concat_label(h3, label))
       
        return h4_mu, h4_logvar 

    def decode(self, z, label):
        
        print(z.shape)
        h5_ = self.upconv1_bn(self.upconv1(self.concat_label(z, label)))
        h5_gated = self.upconv1_gated_bn(self.upconv1(self.concat_label(z, label)))
        h5 = torch.mul(h5_, self.upconv1_sigmoid(h5_gated)) 
        
        print(h5.shape)
        
        h6_ = self.upconv2_bn(self.upconv2(self.concat_label(h5, label)))
        h6_gated = self.upconv2_gated_bn(self.upconv2(self.concat_label(h5, label)))
        h6 = torch.mul(h6_, self.upconv2_sigmoid(h6_gated)) 
        
        print(h6.shape)
        
        h7_ = self.upconv3_bn(self.upconv3(self.concat_label(h6, label)))
        h7_gated = self.upconv3_gated_bn(self.upconv3(self.concat_label(h6, label)))
        h7 = torch.mul(h7_, self.upconv3_sigmoid(h7_gated)) 
        
        print(h7.shape)
        
        h8_mu = self.upconv4_mu(self.concat_label(h7, label))
        h8_logvar = self.upconv4_logvar(self.concat_label(h7, label))
        
        return h8_mu, h8_logvar
    
    def classify(self, x):
        
        h9_ = self.ac_conv1_bn(self.ac_conv1(x))
        h9_gated = self.ac_conv1_gated_bn(self.ac_conv1_gated(x))
        h9 = torch.mul(h9_, self.ac_conv1_sigmoid(h9_gated))
        
        h10_ = self.ac_conv2_bn(self.ac_conv2(h9))
        h10_gated = self.ac_conv2_gated_bn(self.ac_conv2_gated(h9))
        h10 = torch.mul(h10_, self.ac_conv2_sigmoid(h10_gated))
        
        h11_ = self.ac_conv3_bn(self.ac_conv3(h10))
        h11_gated = self.ac_conv3_gated_bn(self.ac_conv3_gated(h10))
        h11 = torch.mul(h11_, self.ac_conv3_sigmoid(h11_gated))
        
        h12_ = self.ac_conv4_bn(self.ac_conv4(h11))
        h12_gated = self.ac_conv4_gated_bn(self.ac_conv4_gated(h11))
        h12 = torch.mul(h12_, self.ac_conv4_sigmoid(h12_gated))
        
        h13_ = F.softmax(self.ac_conv5(h12))
        h13 = torch.prod(h13_, dim=-1, keepdim=True)
        
        return h13.view(-1, self.label_num)
        
    def concat_label(self, x, label):
        shape = x.shape
        label_layer = torch.zeros(shape[0], self.label_num, shape[2], shape[3])
        print(label)
        print(label[0])
        for i in range(len(x)):
            label_layer[i, int(label[i])] = torch.ones(shape[2], shape[3])
        label_layer = label_layer.to(self.device)
        return torch.cat((x, label_layer), dim=1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, label):
        mu_enc, logvar_enc = self.encode(x, label)
        z_enc = self.reparameterize(mu_enc, logvar_enc)
        mu_dec, logvar_dec = self.decode(z_enc, label)
        z_dec = self.reparameterize(mu_dec, logvar_dec)
        p_label = self.classify(z_dec)
        return z_dec, mu_enc, logvar_enc, p_label
                   
    # Reconstruction + KL divergence losses summed over all elements and batch
    def calc_loss(self, x, label):
        
        x = x.to(self.device)
        label = label.to(self.device)

        recon_x, mu, logvar, p_label = self.forward(x, label)
        t_label = self.classify(x)

        # 1
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 2
        AC_1 = F.binary_cross_entropy(label_, p_label) 

        # 3
        AC_2 = F.binary_cross_entropy(label_, t_label) 

        return BCE + KLD + AC_1 + AC_2

    def predict(self, x, label, label_target):
        
        shape = x.shape
        x = x.view(-1, shape[0], shape[1], shape[2])
        x.to(self.device)
        
        mu_enc, logvar_enc = self.encode(x, label)
        z_enc = self.reparameterize(mu_enc, logvar_enc)
        mu_dec, logvar_dec = self.decode(z_enc, label_target)
        z_dec = self.reparameterize(mu_dec, logvar_dec)
        
        return z_dec
    