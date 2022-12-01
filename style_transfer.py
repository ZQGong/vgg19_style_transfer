# %%
# # Image Style Transfer by VGG19, a Deep Transfer Learning Model
#
# This project implements the algorithm found in [(Gatys
# 2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

# %%
# Import libraries used
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as utils

from PIL import Image

# %%
# Define the resizing parameter of the input images
imsize = 128
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

# Load 'content_img' as a torch tensor of size (3*imsize*imsize)
image = Image.open("./data/images/dancing.jpg")
content_img = loader(image)

# Load 'style_img' as a torch tensor of size (3*imsize*imsize)
image = Image.open("./data/images/monet.jpg")
style_img = loader(image)

# %%
# ## Feature extraction with VGG19

class VGG19Features(nn.Module):
    def __init__(self, modules_indexes):
        super(VGG19Features, self).__init__()

        # VGG19 pretrained model in evaluation mode
        self.vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()

        # Indexes of layers to remember
        # which refers to the features set after a certain conv layer
        self.modules_indexes = modules_indexes

    def forward(self, input):
        # Define a hardcoded mean and std deviation of size (3*1*1)
        # 'mean' & 'std' are transformed to 3d tensors with 2 "fake" dimensions of length 1
        # in order that we can apply the broadcasting
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        # First center and normalize 'input' with 'mean' and 'std'
        input_norm = (input - mean) / std

        # Add a fake mini-batch dimension to 'input_norm'
        input_norm = input_norm.unsqueeze(0)

        # Install hooks on specified modules to save their features
        features = []
        handles = []
        for module_index in self.modules_indexes:

            def hook(module, input, output):
                features.append(output)

            handle = self.vgg19.features[module_index].register_forward_hook(hook)
            handles.append(handle)

        # Forward propagate 'input_norm'. This will trigger the hooks
        # set up above and populate 'features'
        self.vgg19(input_norm)

        # Remove hooks
        [handle.remove() for handle in handles]

        # The output of our custom VGG19Features neural network is a
        # list of features of 'inputs'
        return features


# %%
# Define the convolutional layers we will use to capture the style and the content.
#
# Indexes of interesting features to extract
# Define 'modules_indexes'
# According to the wanted layers in 3.2 of the paper on page 2419
modules_indexes = [0, 2, 5, 7, 10]

vgg19 = VGG19Features(modules_indexes)
content_features = [f.detach() for f in vgg19.forward(content_img)]

# %% 
# ## Style features as gram matrix of convolutional features

# Here computes the gram matrix of 'input'. We first need to reshape 'input' 
# before computing the gram matrix.

def gram_matrix(input):
    batchsize, n_filters, width, height = input.size()

    # Reshape 'input' into (n_filters*n_pixels)
    features = input.view(n_filters, width*height)

    # Compute the inner products between filters in 'G'
    G = torch.mm(features, features.t())

    # We normalize the values of the gram matrix by dividing by the
    # number of element in each feature maps.
    return G.div(n_filters * width * height)


style_gram_features = [gram_matrix(f.detach()) for f in vgg19.forward(style_img)]

target = content_img.clone().requires_grad_(True)

# %%
# Define 'optimizer' to use L-BFGS algorithm to do gradient descent on 'target'
optimizer = optim.LBFGS([target])

# %%
# ## The algorithm From the paper, there are two different losses. 
# The style loss and the content loss.

# Define 'style_weight' the trade-off parameter between style and content losses.
# alpha/beta value noted in 3.1 of paper on page 2419 can be used
# but better try other values
style_weight = 1.5*10**6

# %%

for step in range(500):
    # First, forward propagate 'target' through our VGG19Features neural
    # network and store its output as 'target_features'
    target_features = vgg19(target)

    # Define now the 'content_loss' on the first layer only
    content_loss = torch.sum((target_features[0] - content_features[0])**2)

    style_loss = 0
    for target_feature, style_gram_feature in zip(target_features, style_gram_features):
        # Compute Gram matrix
        target_gram_feature = gram_matrix(target_feature)

        # Add current loss to 'style_loss'
        style_loss += torch.sum((target_gram_feature - style_gram_feature)**2)

    # Compute combined loss
    loss = content_loss + style_weight * style_loss

    # Backpropagate gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(lambda: loss)

    if step % 20 == 0:
        print("step {}:".format(step))
        print(
            "Style Loss: {:4f} Content Loss: {:4f} Overall: {:4f}".format(style_loss.item(), content_loss.item(), loss.item())
        )
        img = target.clone().squeeze()
        img = img.clamp_(0, 1)
        utils.save_image(img, "output-{}.png".format(step))

# %%
