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
from utils import save_checkpoint
from PIL import Image


class BatchDict(TypedDict):
    A: torch.Tensor
    B: torch.Tensor


class SetupVars(TypedDict):
    dis_X: Discriminator
    dis_Y: Discriminator
    gen_X: Generator
    gen_Y: Generator
    l1_loss: torch.nn.L1Loss
    mse_loss: torch.nn.MSELoss
    dis_combined_params: list[torch.nn.parameter.Parameter]
    gen_combined_params: list[torch.nn.parameter.Parameter]
    optimizer_D: torch.optim.Adam
    optimizer_G: torch.optim.Adam
    dataset: ImageDataset
    loader: DataLoader[BatchDict]


class ForwardVars(TypedDict):
    real_X: torch.Tensor
    real_Y: torch.Tensor
    fake_X: torch.Tensor
    fake_Y: torch.Tensor
    rec_X: torch.Tensor
    rec_Y: torch.Tensor


def set_requires_grad(nets: list[nn.Module], requires_grad: bool):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def setup() -> SetupVars:
    dis_X = Discriminator().to(config.DEVICE)
    dis_Y = Discriminator().to(config.DEVICE)
    gen_X = Generator().to(config.DEVICE)
    gen_Y = Generator().to(config.DEVICE)

    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()

    dis_combined_params = [*dis_X.parameters(), *dis_Y.parameters()]
    optimizer_D = torch.optim.Adam(
        dis_combined_params, lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    gen_combined_params = [*gen_X.parameters(), *gen_Y.parameters()]
    optimizer_G = torch.optim.Adam(
        gen_combined_params, lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    dataset = ImageDataset(
        root_A=os.path.join(config.TRAIN_DIR, "trainA"),
        root_B=os.path.join(config.TRAIN_DIR, "trainB"),
        transforms=config.input_transforms
    )

    loader = get_loader_from_dataset(dataset)

    return {
        "dis_X": dis_X,
        "dis_Y": dis_Y,
        "gen_X": gen_X,
        "gen_Y": gen_Y,
        "l1_loss": l1_loss,
        "mse_loss": mse_loss,
        "dis_combined_params": dis_combined_params,
        "gen_combined_params": gen_combined_params,
        "optimizer_D": optimizer_D,
        "optimizer_G": optimizer_G,
        "dataset": dataset,
        "loader": loader
    }


def forward(setup_vars: SetupVars, X: torch.Tensor, Y: torch.Tensor) -> ForwardVars:
    real_X = X.to(config.DEVICE)
    real_Y = Y.to(config.DEVICE)

    fake_X = setup_vars["gen_Y"](real_Y)
    fake_Y = setup_vars["gen_X"](real_X)
    rec_X = setup_vars["gen_Y"](fake_Y)
    rec_Y = setup_vars["gen_X"](fake_X)

    return {
        "real_X": real_X,
        "real_Y": real_Y,
        "fake_X": fake_X,
        "fake_Y": fake_Y,
        "rec_X": rec_X,
        "rec_Y": rec_Y,
    }


def _train_discriminator(svars: SetupVars, fvars: ForwardVars):
    D_score_real_X = svars["dis_X"](fvars["real_X"])
    D_score_fake_X = svars["dis_X"](fvars["fake_X"].detach())
    D_score_real_Y = svars["dis_Y"](fvars["real_Y"])
    D_score_fake_Y = svars["dis_Y"](fvars["fake_Y"].detach())

    D_loss_real_X = svars["mse_loss"](
        D_score_real_X,
        torch.ones_like(D_score_real_X).to(config.DEVICE)
    )
    D_loss_fake_X = svars["mse_loss"](
        D_score_fake_X,
        torch.zeros_like(D_score_fake_X).to(config.DEVICE)
    )
    D_loss_real_Y = svars["mse_loss"](
        D_score_real_Y,
        torch.ones_like(D_score_real_Y).to(config.DEVICE)
    )
    D_loss_fake_Y = svars["mse_loss"](
        D_score_fake_Y,
        torch.zeros_like(D_score_fake_Y).to(config.DEVICE)
    )

    D_loss_X = (D_loss_real_X + D_loss_fake_X) / 2
    D_loss_Y = (D_loss_real_Y + D_loss_fake_Y) / 2
    D_loss = D_loss_X + D_loss_Y

    return D_loss


def _train_generator(svars: SetupVars, fvars: ForwardVars):
    val_G_X = svars["dis_X"](fvars["fake_X"])
    adv_loss_G_X = svars["mse_loss"](
        val_G_X,
        torch.ones_like(val_G_X).to(config.DEVICE)
    )

    val_G_Y = svars["dis_Y"](fvars["fake_Y"])
    adv_loss_G_Y = svars["mse_loss"](
        val_G_Y,
        torch.ones_like(val_G_Y).to(config.DEVICE)
    )

    cycle_loss_G_X = config.LAMBDA_CYCLE * svars["l1_loss"](
        fvars["rec_X"], fvars["real_X"]
    )
    cycle_loss_G_Y = config.LAMBDA_CYCLE * svars["l1_loss"](
        fvars["rec_Y"], fvars["real_Y"]
    )

    iden_X = svars["gen_Y"](fvars["real_X"])
    iden_Y = svars["gen_X"](fvars["real_Y"])

    # TODO: Not sure if config.LAMBDA_CYCLE is needed in the identity loss.
    # TODO: What is lambda A and lambda B? Because config only has LAMBDA_IDENTITY.
    iden_loss_X = config.LAMBDA_IDENTITY * svars["l1_loss"](
        iden_X, fvars["real_X"]
    )
    iden_loss_Y = config.LAMBDA_IDENTITY * svars["l1_loss"](
        iden_Y, fvars["real_Y"]
    )

    loss_G = adv_loss_G_X + adv_loss_G_Y + \
        cycle_loss_G_X + cycle_loss_G_Y + \
        iden_loss_X + iden_loss_Y

    return loss_G


def train_discriminator(svars: SetupVars, fvars: ForwardVars):
    set_requires_grad([svars["gen_X"], svars["gen_Y"]], False)

    svars["optimizer_D"].zero_grad()

    D_loss = _train_discriminator(svars, fvars)
    D_loss.backward()
    svars["optimizer_D"].step()

    set_requires_grad([svars["gen_X"], svars["gen_Y"]], True)


def train_generator(svars: SetupVars, fvars: ForwardVars):
    set_requires_grad([svars["dis_X"], svars["dis_Y"]], False)

    svars["optimizer_G"].zero_grad()

    G_loss = _train_generator(svars, fvars)
    G_loss.backward()
    svars["optimizer_G"].step()

    set_requires_grad([svars["dis_X"], svars["dis_Y"]], True)


def train(svars: SetupVars):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config.RUNS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    for i in range(config.NUM_EPOCHS):
        for j, sample in enumerate(tqdm(svars["loader"])):
            X, Y = sample["A"], sample["B"]

            fvars = forward(svars, X, Y)
            train_generator(svars, fvars)
            train_discriminator(svars, fvars)

            if config.SAVE_MODEL and j % config.CHECKPOINT_FREQ == 0:
                try:
                    svars["gen_X"].eval()
                    svars["gen_Y"].eval()
                    svars["dis_X"].eval()
                    svars["dis_Y"].eval()

                    save_checkpoint(
                        svars["gen_X"],
                        os.path.join(run_dir, config.CHECKPOINT_GEN_A)
                    )
                    save_checkpoint(
                        svars["gen_Y"],
                        os.path.join(run_dir, config.CHECKPOINT_GEN_B)
                    )
                    save_checkpoint(
                        svars["dis_X"],
                        os.path.join(run_dir, config.CHECKPOINT_CRITIC_A)
                    )
                    save_checkpoint(
                        svars["dis_Y"],
                        os.path.join(run_dir, config.CHECKPOINT_CRITIC_B)
                    )

                    with torch.no_grad():
                        image_fake_X: Image.Image = config.output_transforms(
                            fvars["fake_X"].detach()[0]
                        )  # type: ignore
                        image_fake_Y: Image.Image = config.output_transforms(
                            fvars["fake_Y"].detach()[0]
                        )  # type: ignore

                        image_fake_X.save(os.path.join(
                            run_dir, f"{i}_{j}_fake_X.png"))
                        image_fake_Y.save(os.path.join(
                            run_dir, f"{i}_{j}_fake_Y.png"))
                finally:
                    svars["gen_X"].train()
                    svars["gen_Y"].train()
                    svars["dis_X"].train()
                    svars["dis_Y"].train()


def main():
    setup_vars = setup()
    train(setup_vars)


if __name__ == "__main__":
    main()
