import torch
import torch.nn as nn
import torchvision as tv
from torchvision import transforms
from torch.util.data import ImageFolder, DataLoader

# get dataset
def download(source, target, force_clear=False):
    if source.startwith('http'):
        wget.download(source,target)
        print('done and dusted')

def tf(path):
    img = tv.io.read_image(path)
    img = tv.io.decode_jpg(img, 3) # here this represents 3 channels ( RGB )
    return img

# load dataset
def load_dataset(source, target, force_clear=False):
    download(source, target, force_clear)
    dataset = ImageFolder(target, transform=tf)
    return dataset

# load dataloader
def load_dataloader(source, target, force_clear=False):
    dataset = load_dataset(source, target, force_clear)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    return dataloader

# load model
def load_model():
    model = tv.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    return model

# train model
def train_model(model, dataloader, epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return model

# save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# load model
def load_model(path):
    model = tv.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(path))
    return model

# test model
def test_model(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
        
