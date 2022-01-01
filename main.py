import torch
import torchvision.datasets as dataset
from torch.utils.data import DataLoader

from src.dataset import SiameseDataset
import torchvision.transforms as transforms
from src.configs import TrainingConfigs
from src.model import SiameseNetwork, ConstrastiveLoss
from torch import optim
import matplotlib.pyplot as plt
import os


def main():
    image_folder = dataset.ImageFolder(root=TrainingConfigs.train_dir)
    siamese_dataset = SiameseDataset(image_folder=image_folder,
                                     transform=transforms.Compose(
                                         [transforms.Resize((250, 250)), transforms.ToTensor()]),
                                     invert=False, showimage=False)

    dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=TrainingConfigs.num_of_workers,
                            batch_size=TrainingConfigs.train_batch_size)
    if torch.cuda.is_available():
        model = SiameseNetwork().cuda()
    else:
        model = SiameseNetwork()

    criterion = ConstrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration = 0
    print("Training is started")
    for epoch in range(TrainingConfigs.train_num_epochs):
        for i, data in enumerate(dataloader, 0):
            input1, input2, label = data
            if torch.cuda.is_available():
                input1, input2, label = input1.cuda(), input2.cuda(), label.cuda()
            else:
                input1, input2, label = input1, input2, label
            optimizer.zero_grad()
            output1, output2 = model(input1, input2)
            contrastive_loss = criterion(output1, output2, label)
            contrastive_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Epoch: {} \t Loss: {}".format(epoch, contrastive_loss.item()))
                iteration += 10
                loss_history.append(contrastive_loss.item())
                counter.append(iteration)

    plt.plot(counter, loss_history)
    plt.show()


if __name__ == '__main__':
    main()
