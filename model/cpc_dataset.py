from PIL import Image
import torch
from torch.utils.data import Dataset


class CPCDataset_Multimodal(Dataset):
    def __init__(self, images_path1: list, images_path2: list, images_class: list, transform=None):
        self.images_path1 = images_path1
        self.images_path2 = images_path2
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path1)

    def __getitem__(self, item):
        img1 = Image.open(self.images_path1[item])
        img2 = Image.open(self.images_path2[item])
        if img1.mode != 'RGB':
            raise ValueError("image1: {} isn't RGB mode.".format(self.images_path1[item]))
        if img2.mode != 'RGB':
            raise ValueError("image2: {} isn't RGB mode.".format(self.images_path2[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    @staticmethod
    def collate_fn(batch):
        images1, images2, labels = tuple(zip(*batch))
        images1 = torch.stack(images1, dim=0)
        images2 = torch.stack(images2, dim=0)
        labels = torch.as_tensor(labels)
        return images1, images2, labels
