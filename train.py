from typing import TypedDict
from dataset_loader import ImageDataset, get_loader_from_dataset
from models.discriminator import Discriminator
from models.generator import Generator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
import os
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import make_grid, save_image

class ReplayBuffer:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        to_return = []

        for img in data.detach().cpu():
            img = img.unsqueeze(0) # (1, C, H, W)

            if len(self.data) < self.max_size:
                self.data.append(img)
                to_return.append(img)
            else:
                # Return old and replace with new
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, len(self.data), (1,)).item()
                    old = self.data[idx]
                    self.data[idx] = img
                    to_return.append(old)
                # Return new
                else:
                    to_return.append(img)
        
        return torch.cat(to_return, dim=0).to(config.DEVICE)


class BatchDict(TypedDict):
    A: torch.Tensor
    B: torch.Tensor


class SetupVars(TypedDict):
    D_A: Discriminator
    D_B: Discriminator
    G_AB: Generator
    G_BA: Generator
    l1_loss: torch.nn.L1Loss
    mse_loss: torch.nn.MSELoss
    optim_D_A: torch.optim.Adam
    optim_D_B: torch.optim.Adam
    optim_G: torch.optim.Adam
    scheduler_G: torch.optim.lr_scheduler.LambdaLR
    scheduler_D_A: torch.optim.lr_scheduler.LambdaLR
    scheduler_D_B: torch.optim.lr_scheduler.LambdaLR
    fake_A_buffer: ReplayBuffer
    fake_B_buffer: ReplayBuffer
    dataset: ImageDataset
    loader: DataLoader[BatchDict]
    start_epoch: int


class ForwardVars(TypedDict):
    fake_A: torch.Tensor
    fake_B: torch.Tensor
    rec_A: torch.Tensor
    rec_B: torch.Tensor


def save_checkpoint_all(run_ckpt_dir, epoch, G_AB, G_BA, D_A, D_B, optim_G, optim_D_A, optim_D_B, scheduler_G=None, scheduler_D_A=None, scheduler_D_B=None):
    os.makedirs(run_ckpt_dir, exist_ok=True)
    filename_checkpoint = os.path.join(run_ckpt_dir, f"checkpoint_{epoch}.pth")
    filename_checkpoint_latest = os.path.join(run_ckpt_dir, f"latest_checkpoint.pth")
    filename_latest = os.path.join(run_ckpt_dir, "latest.txt")
    
    to_save = {
        "epoch": epoch,
        "G_AB": G_AB.state_dict(),
        "G_BA": G_BA.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "optim_G": optim_G.state_dict(),
        "optim_D_A": optim_D_A.state_dict(),
        "optim_D_B": optim_D_B.state_dict(),
        "scheduler_G": scheduler_G.state_dict() if scheduler_G is not None else None,
        "scheduler_D_A": scheduler_D_A.state_dict() if scheduler_D_A is not None else None,
        "scheduler_D_B": scheduler_D_B.state_dict() if scheduler_D_B is not None else None
    }

    torch.save(to_save, filename_checkpoint)
    torch.save(to_save, filename_checkpoint_latest)

    with open(filename_latest, "w") as f:
        f.write(filename_checkpoint_latest)

def load_checkpoint_all(ckpt_file_path, G_AB, G_BA, D_A, D_B, optim_G, optim_D_A, optim_D_B, scheduler_G=None, scheduler_D_A=None, scheduler_D_B=None, device="cpu"):
    ckpt = torch.load(ckpt_file_path, map_location=device)

    if "G_AB" in ckpt:
        G_AB.load_state_dict(ckpt["G_AB"])
        G_BA.load_state_dict(ckpt["G_BA"])
        D_A.load_state_dict(ckpt["D_A"])
        D_B.load_state_dict(ckpt["D_B"])

    if "optim_G" in ckpt:
        optim_G.load_state_dict(ckpt["optim_G"])
        optim_D_A.load_state_dict(ckpt["optim_D_A"])
        optim_D_B.load_state_dict(ckpt["optim_D_B"])

    if scheduler_G is not None and ckpt.get("scheduler_G", None) is not None:
        scheduler_G.load_state_dict(ckpt["scheduler_G"])
        scheduler_D_A.load_state_dict(ckpt["scheduler_D_A"])
        scheduler_D_B.load_state_dict(ckpt["scheduler_D_B"])

    return ckpt.get("epoch", None)


def set_requires_grad(nets: list[nn.Module], requires_grad: bool):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def setup() -> SetupVars:
    G_AB = Generator().to(config.DEVICE) # G_AB = A -> B
    G_BA = Generator().to(config.DEVICE) # G_BA = B -> A
    D_A = Discriminator().to(config.DEVICE)  # D_A = critique A
    D_B = Discriminator().to(config.DEVICE)  # D_B = critique B

    G_AB.to(config.DEVICE)
    G_BA.to(config.DEVICE)
    D_A.to(config.DEVICE)
    D_B.to(config.DEVICE)

    l1_loss = torch.nn.L1Loss().to(config.DEVICE)
    mse_loss = torch.nn.MSELoss().to(config.DEVICE)

    optim_D_A = torch.optim.Adam(
        D_A.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    optim_D_B = torch.optim.Adam(
        D_B.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    gen_combined_params = list(G_AB.parameters()) + list(G_BA.parameters())
    optim_G = torch.optim.Adam(
        gen_combined_params, lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    def lr_lambda(epoch):
        if epoch < config.DECAY_START:
            return 1.0
        else:
            epoch_normalized = (epoch - config.DECAY_START) / float(config.NUM_EPOCHS - config.DECAY_START)
            return max(0.0, 1.0 - epoch_normalized)
    
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optim_D_A, lr_lambda)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optim_D_B, lr_lambda)

    dataset = ImageDataset(
        root_A=os.path.join(config.TRAIN_DIR, "trainA"),
        root_B=os.path.join(config.TRAIN_DIR, "trainB"),
        transforms=config.input_transforms
    )

    loader = get_loader_from_dataset(dataset)

    fake_A_buffer = ReplayBuffer(max_size=config.REPLAY_BUFFER_SIZE)
    fake_B_buffer = ReplayBuffer(max_size=config.REPLAY_BUFFER_SIZE)

    start_epoch = 0
    if config.RESUME_TRAIN:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_{config.RESUME_EPOCH}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found")
        
        ckpt_epoch = load_checkpoint_all(ckpt_path, G_AB, G_BA, D_A, D_B, optim_G, optim_D_A, optim_D_B, scheduler_G, scheduler_D_A, scheduler_D_B, config.DEVICE)
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        if ckpt_epoch is None:
            start_epoch = config.RESUME_EPOCH + 1
            print(f"Resuming training from epoch index of {config.RESUME_EPOCH}. Next epoch will start at epoch index of {start_epoch}. Using config.RESUME_EPOCH = {config.RESUME_EPOCH}")
        else:
            start_epoch = int(ckpt_epoch) + 1
            print(f"Resuming training from epoch index of {config.RESUME_EPOCH}. Next epoch will start at epoch index of {start_epoch}. Using ckpt_epoch = {ckpt_epoch}")

    return {
        "D_A": D_A,
        "D_B": D_B,
        "G_AB": G_AB,
        "G_BA": G_BA,
        "l1_loss": l1_loss,
        "mse_loss": mse_loss,
        "optim_D_A": optim_D_A,
        "optim_D_B": optim_D_B,
        "optim_G": optim_G,
        "scheduler_G": scheduler_G,
        "scheduler_D_A": scheduler_D_A,
        "scheduler_D_B": scheduler_D_B,
        "fake_A_buffer": fake_A_buffer,
        "fake_B_buffer": fake_B_buffer,
        "dataset": dataset,
        "loader": loader,
        "start_epoch": start_epoch
    }


def forward(G_AB: Generator, G_BA: Generator, real_A: torch.Tensor, real_B: torch.Tensor) -> ForwardVars:
    real_A = real_A.to(config.DEVICE)
    real_B = real_B.to(config.DEVICE)

    fake_A = G_BA(real_B) # G_BA(B) -> A
    fake_B = G_AB(real_A) # G_AB(A) -> B
    rec_A = G_BA(fake_B)  # G_BA(G_AB(A)) -> A
    rec_B = G_AB(fake_A)  # G_AB(G_BA(B)) -> B

    return {
        "fake_A": fake_A,
        "fake_B": fake_B,
        "rec_A": rec_A,
        "rec_B": rec_B,
    }

def train_epoch(svars: SetupVars, epoch_index):
    G_AB, G_BA, D_A, D_B = svars["G_AB"], svars["G_BA"], svars["D_A"], svars["D_B"]
    optim_G, optim_D_A, optim_D_B = svars["optim_G"], svars["optim_D_A"], svars["optim_D_B"]
    fake_A_buffer = svars["fake_A_buffer"]
    fake_B_buffer = svars["fake_B_buffer"]
    mse_loss, l1_loss = svars["mse_loss"], svars["l1_loss"]
    loader = svars["loader"]
    
    for batch in tqdm(loader, desc=f"Epoch {epoch_index+1}/{config.NUM_EPOCHS}"):
        real_A = batch["A"].to(config.DEVICE) # Vangogh (peintre)
        real_B = batch["B"].to(config.DEVICE) # Photos

        ##### Generators -----
        set_requires_grad([D_A, D_B], False)
        optim_G.zero_grad()

        forwardVars = forward(G_AB, G_BA, real_A, real_B)
        fake_A, fake_B, rec_A, rec_B = forwardVars["fake_A"], forwardVars["fake_B"], forwardVars["rec_A"], forwardVars["rec_B"]

        # Adversarial losses
        pred_fake_A = D_A(fake_A) # G_BA(B) -> A
        pred_fake_B = D_B(fake_B) # G_AB(A) -> B
        adv_loss_A = mse_loss(pred_fake_A, torch.ones_like(pred_fake_A).to(config.DEVICE))
        adv_loss_B = mse_loss(pred_fake_B, torch.ones_like(pred_fake_B).to(config.DEVICE))
        adv_loss = (adv_loss_A + adv_loss_B) * 0.5

        # Cycle losses
        cycle_loss_A = l1_loss(rec_A, real_A) # G_BA(G_AB(A)) & A
        cycle_loss_B = l1_loss(rec_B, real_B) # G_AB(G_BA(B)) & B
        cycle_loss = config.LAMBDA_CYCLE * (cycle_loss_A + cycle_loss_B) * 0.5

        # Identity loss
        if config.USE_IDENTITY:
            idt_A = G_BA(real_A) # G_BA(A) -> A
            idt_B = G_AB(real_B) # G_AB(B) -> B
            id_loss_A = l1_loss(idt_A, real_A)
            id_loss_B = l1_loss(idt_B, real_B)
            id_loss = config.LAMBDA_IDENTITY * (id_loss_A + id_loss_B) * 0.5
        else:
            id_loss = torch.tensor(0.0, device=config.DEVICE)

        loss_G = adv_loss + cycle_loss + id_loss
        loss_G.backward()
        optim_G.step()
        set_requires_grad([D_A, D_B], True)

        ##### Discriminator A -----
        optim_D_A.zero_grad()
        pred_real_A = D_A(real_A)
        loss_D_A_real = mse_loss(pred_real_A, torch.ones_like(pred_real_A).to(config.DEVICE))

        fake_A_for_D = fake_A_buffer.push_and_pop(fake_A)
        pred_fake_A = D_A(fake_A_for_D.detach())
        loss_D_A_fake = mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A).to(config.DEVICE))

        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()
        optim_D_A.step()
       
        ##### Discriminator B -----
        optim_D_B.zero_grad()
        pred_real_B = D_B(real_B)
        loss_D_B_real = mse_loss(pred_real_B, torch.ones_like(pred_real_B).to(config.DEVICE))

        fake_B_for_D = fake_B_buffer.push_and_pop(fake_B)
        pred_fake_B = D_B(fake_B_for_D.detach())
        loss_D_B_fake = mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B).to(config.DEVICE))

        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward()
        optim_D_B.step()



def train():
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config.RUNS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    svars = setup()
    G_AB, G_BA, D_A, D_B = svars["G_AB"], svars["G_BA"], svars["D_A"], svars["D_B"]
    scheduler_G, scheduler_D_A, scheduler_D_B = svars["scheduler_G"], svars["scheduler_D_A"], svars["scheduler_D_B"]
    optim_G, optim_D_A, optim_D_B = svars["optim_G"], svars["optim_D_A"], svars["optim_D_B"]

    start_epoch = svars["start_epoch"]
    for epoch_index in range(start_epoch, config.NUM_EPOCHS):
        train_epoch(svars, epoch_index)

        # Steping the schedulers
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        if epoch_index % config.SAMPLE_EVERY == 0 or epoch_index == config.NUM_EPOCHS - 1:
            batch = next(iter(svars["loader"]))
            real_A = batch["A"].to(config.DEVICE)
            real_B = batch["B"].to(config.DEVICE)
            with torch.no_grad():
                fake_A = svars["G_BA"](real_B)
                fake_B = svars["G_AB"](real_A)

            grid = make_grid(torch.cat([real_A, fake_B, real_B, fake_A], dim=0), nrow=real_A.size(0))
            save_image((grid * 0.5) + 0.5, os.path.join(sample_dir, f"epoch_{epoch_index}.png"))

        # Saving checkpoints
        if config.SAVE_MODEL and (epoch_index % config.CKPT_EVERY == 0 or epoch_index == config.NUM_EPOCHS - 1):
            save_checkpoint_all(ckpt_dir, epoch_index, G_AB, G_BA, D_A, D_B, optim_G, optim_D_A, optim_D_B, scheduler_G, scheduler_D_A, scheduler_D_B)

if __name__ == "__main__":
    train()
