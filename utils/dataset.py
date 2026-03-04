import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self,train=True):
        self.transform=transforms.Compose([transforms.ToTensor()])

        self.dataset=datasets.MNIST(root="./data",train=train,download=True,
                                    transform=self.transform)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image,label=self.dataset[index]
        return image,label
    
def get_dataloaders(batch_size=32):
    train_dataset=MNISTDataset(train=True)
    test_dataset=MNISTDataset(train=False)

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader,test_loader
