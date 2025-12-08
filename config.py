from enum import Enum
import torch
import torchvision.transforms as transforms

class Implementation(Enum):
    CUSTOM = "custom"
    PAPER = "paper"
    RP = "rp"

IMPLEMENTATION = Implementation.CUSTOM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/vangogh2photo/vangogh2photo/"
RUNS_DIR = "runs"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 40
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FREQ = 400
CHECKPOINT_GEN_A = "genh.pth.tar"
CHECKPOINT_GEN_B = "genz.pth.tar"
CHECKPOINT_CRITIC_A = "critich.pth.tar"
CHECKPOINT_CRITIC_B = "criticz.pth.tar"
IMAGE_SIZE = 256

input_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

output_transforms = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    transforms.ToPILImage()
])
