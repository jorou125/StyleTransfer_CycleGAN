import torch
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/vangogh2photo/vangogh2photo/"
RUNS_DIR = "runs"

RESUME_TRAIN = False
CHECKPOINT_DIR = "runs/<replace-with-id>/checkpoints"
RESUME_EPOCH = 0

SEED = 42
REPLAY_BUFFER_SIZE = 50
NUM_EPOCHS = 100
DECAY_START = NUM_EPOCHS + 1 # No decay since only doing 100 epochs and not 200
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
USE_IDENTITY = True
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
LOAD_MODEL = False
SAVE_MODEL = True
SAMPLE_EVERY = 1
CKPT_EVERY = 4
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
