import torch as th
import torchvision as thv
from NN_def import NN
import sys
from PIL import Image

if __name__=="__main__":
  path_to_pth_models = "../../static_res/pthModels/"

  device = "cuda" if th.cuda.is_available() else "cpu"
  print(f"using {device} device")

  model = NN()
  model.load_state_dict(th.load(path_to_pth_models + "fashion_MNIST_model.pth"))
  model.to(device)

  path = sys.argv[1]
  pred = None
  with Image.open(path) as img:
    transform = thv.transforms.ToTensor()
    imgTensor = transform(img)
    imgTensor.to(device)
    model.eval()
    pred = model(imgTensor)

  print(pred)