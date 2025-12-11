import torchvision
import torchvision.transforms as transforms

# Define a transformation to convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the MNIST training dataset to data directory, one level up

train_dataset = torchvision.datasets.MNIST(root='../data',
                                           train=True,
                                           transform=transform,
                                           download=True)

# Download and load the MNIST test dataset
test_dataset = torchvision.datasets.MNIST(root='../data',
                                          train=False,
                                          transform=transform,
                                          download=True)