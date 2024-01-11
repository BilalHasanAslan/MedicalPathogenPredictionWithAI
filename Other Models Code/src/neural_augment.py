from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np

'''
This code implements the Neural Style Augmentation method described by Gatys et al.
We use the Neural Augmentation code by publically available at https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/neuralstyle
This code has been modified to fit our training pipeline
'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path_or_tensor, max_size=None, shape=None):
    """Load an image and convert it to a torch tensor."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])
    try:
        extension = image_path_or_tensor[:-3]
        image = Image.open(image_path_or_tensor)  
    except:
        image = image_path_or_tensor
        
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)
    
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    
    # Apply ToTensor transform and move to device
    image = transform(image).to(device)
    
    return image

class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28'] 
        self.vgg = models.vgg19(pretrained=True).features
        
    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

def neural_style(content, style_image_path, max_size=400, total_step=2000, log_step=10, sample_step=500, style_weight=100):
    # Load content and style images
    # Make sure content and style are 4D tensors (batch_size, num_channels, height, width)
    if torch.is_tensor(content):
        content = content.unsqueeze(0)
    else:
        content = load_image(content, max_size=max_size)

    style = load_image(style_image_path, shape=[content.size(1), content.size(2)])

    target = content.clone()

    optimizer = torch.optim.Adam([target.requires_grad_()], lr=0.003, betas=[0.5, 0.999])
    vgg = VGGNet().to(device).eval()

    for step in range(total_step):
        target_features = vgg(target.unsqueeze(0))  # add batch dimension
        content_features = vgg(content.unsqueeze(0))  # add batch dimension
        style_features = vgg(style.unsqueeze(0))  # add batch dimension
        #print(step)

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            # Compute content loss with target and content images
            content_loss += torch.mean((f1 - f2)**2)

            # Reshape convolutional feature maps
            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            # Compute gram matrix
            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())

            # Compute style loss with target and style images
            style_loss += torch.mean((f1 - f3)**2) / (c * h * w) 
        
        # Compute total loss, backprop and optimize
        loss = content_loss + style_weight * style_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the generated image
    img = target.clone().squeeze()
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = denorm(img).clamp_(0, 1)
    
    # Convert the tensor image to PIL Image to be compatible with other torchvision transforms
    img = transforms.ToPILImage()(img.cpu())

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='png/content.png')
    parser.add_argument('--style', type=str, default='png/style.png')
    parser.add_argument('--max_size', type=int, default=400)
    parser.add_argument('--total_step', type=int, default=500)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=125)
    parser.add_argument('--style_weight', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    config = parser.parse_args()
    neural_style(config)