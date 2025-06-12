# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Imports
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# 3. Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(images_dir))
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx].replace('.jpg', '.png'))
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = np.array(mask, dtype=np.int64)
        return image, torch.from_numpy(mask)

# 4. Paths and transforms
images_dir = '/content/drive/MyDrive/segmentation_data/images'
masks_dir = '/content/drive/MyDrive/segmentation_data/masks'

transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
mask_transform = transforms.Compose([
    transforms.Resize((520, 520), interpolation=Image.NEAREST)
])

dataset = SegmentationDataset(images_dir, masks_dir, transform, mask_transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

# 5. Model setup
NUM_CLASSES = 2  # background and wall
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None, num_classes=NUM_CLASSES)
model.backbone.load_state_dict(
    torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT).state_dict()
)
model = model.to(device)

# 6. Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 7. Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# 8. Save the model
torch.save(model.state_dict(), '/content/drive/MyDrive/deeplabv3_wall.pth')
print("Model saved to Google Drive!")