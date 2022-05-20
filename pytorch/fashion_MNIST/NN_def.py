import torch as th
import torchvision as thv
import sys

class NN(th.nn.Module):
    def __init__(self):
      super(NN, self).__init__()
      self.flatten = th.nn.Flatten()
      self.linear_relu_stack = th.nn.Sequential(
        th.nn.Linear(28*28, 512),
        th.nn.ReLU(),
        th.nn.Linear(512, 512),
        th.nn.ReLU(),
        th.nn.Linear(512, 10)
      )

    def forward(self, x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits

def get_input_size(path_to_model_dir):
  dummy_data = thv.datasets.FashionMNIST(
    root=path_to_model_dir + "/data",
    train=False,
    download=False,
    transform=thv.transforms.ToTensor()
  )

  X, _ = dummy_data[0]
  
  return X.size()


