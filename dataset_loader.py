from PIL import Image
import os
import config
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, direction="AtoB", transforms=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transforms = transforms
        self.direction = direction

        self.images_A = os.listdir(root_A)
        self.images_B = os.listdir(root_B)

    def __len__(self):
        return max(len(self.images_A), len(self.images_B))
    
    def __getitem__(self, index):
        image_A = self.images_A[index % len(self.images_A)]
        image_B = self.images_B[index % len(self.images_B)]
        path_A = os.path.join(self.root_A, image_A)
        path_B = os.path.join(self.root_B, image_B)
        img_A = Image.open(path_A).convert("RGB")
        img_B = Image.open(path_B).convert("RGB")
        if self.transforms is not None:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)
        return {"A": img_A, "B": img_B}
    
if __name__ == "__main__":
    dataset = ImageDataset(
        root_A="data\\vangogh2photo\\trainA",
        root_B="data\\vangogh2photo\\trainB",
        transforms=config.input_transforms
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    # print(f"Image A shape: {sample['A'].shape}")
    # print(f"Image B shape: {sample['B'].shape}")
    # img_A = config.output_transforms(sample['A'])
    # img_B = config.output_transforms(sample['B'])
    # img_A.show(title="Image A")
    # img_B.show(title="Image B")


def get_loader_from_dataset(dataset):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        # drop_last=True
        # pin_memory=True
    )
