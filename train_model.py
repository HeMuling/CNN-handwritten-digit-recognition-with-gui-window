import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os

# define training function
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def dl_train(net,trainer , train_iter, test_iter, num_epoch, device=None, pre_trained=False):

    device = torch.device('cpu') if device == None else device

    if pre_trained == False:
        net.apply(init_weights)
    
    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    # begin to train
    net.train()
    for epoch in range(num_epoch):
        running_loss = 0.0
        for x, y in enumerate(train_iter):
            data, label = y
            data = data.to(device)
            label = label.to(device)

            trainer.zero_grad()

            y_hat = net(data)
            loss = loss_function(y_hat, label)
            loss.backward()

            trainer.step()

            running_loss += loss.item()
            if x % 100 == 99:
                print(f'epoch {epoch+1}, batch {x+1}, loss {running_loss/100:.3f}')
                running_loss = 0.0
    
    if test_iter != None:
        # begin evaluation
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in enumerate(test_iter):
                data, label = y
                data = data.to(device)
                label = label.to(device)

                y_hat = net(data)
                _, predicted = torch.max(y_hat.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            print(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')

# define network: VGG block
def VGG_block(num_convs, in_channels, out_channels):
    blk = []
    for _ in range(num_convs):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    return nn.Sequential(*blk)

def VGG(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(VGG_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

# load dataset
path = os.path.join('MNIST_data')
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
])

MNIST_train = torchvision.datasets.MNIST(
    path, train=True,
    transform=trans,
    download=True
)
MNIST_test = torchvision.datasets.MNIST(
    path, train=False,
    transform=trans,
    download=True
)

train_iter = torch.utils.data.DataLoader(
    MNIST_train, batch_size=256, 
    shuffle=True, num_workers=4, 
    prefetch_factor=4, pin_memory=True
)
test_iter = torch.utils.data.DataLoader(
    MNIST_test, batch_size=256, 
    shuffle=True, num_workers=4, 
    prefetch_factor=4, pin_memory=True
)

# begin to train
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = VGG(conv_arch)

trainer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4)

device = torch.device('cuda')

num_epoch = 40

dl_train(net, trainer, train_iter, test_iter, num_epoch, device)

# save the trained model
path = os.path.join('VGG_checkpoint.pt')

torch.save({
            'epoch': num_epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': trainer.state_dict(),
            }, path)