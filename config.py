import torch
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/van_gogh2photo/"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_A = "genh.pth.tar"
CHECKPOINT_GEN_B = "genz.pth.tar"
CHECKPOINT_CRITIC_A = "critich.pth.tar"
CHECKPOINT_CRITIC_B = "criticz.pth.tar"
IMAGE_SIZE = 256

transformes = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.ToTensor()
])
