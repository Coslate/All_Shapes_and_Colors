import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import ast

class ShapesColorsDataset(Dataset):
    def __init__(self, csv_file, root_dir, mode='train', transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_rel_path = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_rel_path)
        image = Image.open(img_path).convert("RGB")

        if self.mode == 'test':
            return self.transform(image), img_rel_path  # test mode

        labels = ast.literal_eval(self.annotations.iloc[idx, 1])  # list of (shape, color)
        target_vector = self.encode_labels(labels)
        return self.transform(image), torch.tensor(target_vector).float()        

    def encode_labels(self, labels):
        # 3 shapes × 3 colors = 9 possible (shape, color) combos
        classes = [
            ('circle', 'red'), ('circle', 'green'), ('circle', 'blue'),
            ('square', 'red'), ('square', 'green'), ('square', 'blue'),
            ('triangle', 'red'), ('triangle', 'green'), ('triangle', 'blue')
        ]
        target = [0] * len(classes)
        for label in labels:
            if label in classes:
                target[classes.index(label)] = 1
        return target