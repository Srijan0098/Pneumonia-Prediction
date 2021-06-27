
import torch
import torchvision
from torchvision import utils
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_images(dataloader): 
  for images, labels in dataloader:
    fig, ax = plt.subplots(figsize=(16,16))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=5).permute(1, 2, 0))
    break

def accuracy(output, target):
  output = torch.exp(output)
  top_p,top_class = output.topk(1,dim = 1)
  equals = top_class == target.view(*top_class.shape)
  return torch.mean(equals.type(torch.FloatTensor))
