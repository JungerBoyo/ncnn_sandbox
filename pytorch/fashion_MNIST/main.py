import torch as th
import torchvision as thv
from NN_def import NN

if __name__ == "__main__":
  ### GET DATASETS ###
  downloaded = True

  train_data = thv.datasets.FashionMNIST(
    root="data",
    train=True,
    download=not downloaded,
    transform=thv.transforms.ToTensor()
  )

  test_data = thv.datasets.FashionMNIST(
    root="data",
    train=False,
    download=not downloaded,
    transform=thv.transforms.ToTensor()
  )

  ### DEFINE DATALOADERS ###
  batch_size = 64

  train_loader = th.utils.data.DataLoader(dataset=train_data, batch_size=batch_size)
  test_loader = th.utils.data.DataLoader(dataset=test_data, batch_size=batch_size)

  X, y = next(iter(test_loader))
  print(f"Shape of X: {X.shape}")
  print(f"SHape of y: {y.shape}, {y.dtype}")

  device = "cuda" if th.cuda.is_available() else "cpu"
  print(f"using {device} device")

  model = NN()
  model.to(device)

  ### PARAMS ###
  loss_fn = th.nn.CrossEntropyLoss()
  optimizer = th.optim.SGD(model.parameters(), lr=1e-3)

  def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      pred = model(X)
      loss = loss_fn(pred, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch % len(X) == 0:
        loss, i = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f} [{i:>5d}/{size:>5d}]")
  
  def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with th.no_grad():
      for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(th.float).sum().item()
        test_loss /= num_batches
        correct /= size

  ### TRAINING ###
  epochs = 5
  for t in range(epochs):
    print(f"epoch {t}\n")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)

  ### SAVING ###
  path_to_pth_models_dir = "../static_res/pthModels/"

  th.save(model.state_dict(),  path_to_pth_models_dir + "fashion_MNIST_model.pth")

  ## LOADING
  ## model = i.e NN()
  ## model.load.state_dict(th.load("..."))