import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

class VirtualKITTI2GroupedDataset(Dataset):
    def __init__(self, root_dir, conditions, transform=None):
        self.root_dir = root_dir
        self.conditions = conditions
        self.transform = transform
        self.scene_files = sorted(os.listdir(os.path.join(root_dir, "labels")))

    def __len__(self):
        return len(self.scene_files)

    def __getitem__(self, idx):
        scene_file = self.scene_files[idx]
        label_path = os.path.join(self.root_dir, "labels", scene_file)

        # Load images for all conditions
        images = []
        for condition in self.conditions:
            img_path = os.path.join(self.root_dir, condition, scene_file)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        label = Image.open(label_path)
        if self.transform:
            label = transforms.Resize((512, 1024))(label)
            label = torch.tensor(np.array(label), dtype=torch.long)
        
        return {"images": images, "label": label}

# Example transformations
transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor()
])
