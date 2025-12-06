from PIL import Image
from torchvision import transforms
from models.generator import Generator
import functools
import torch
import torch.nn as nn
import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk

def test_generator_on_image(image_path, generator, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_image = generator(input_image)
    return output_image.cpu().squeeze(0)

def load_pretrained_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen = Generator(input_nc=3, output_nc=3, ngf=64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True), n_blocks=9)
    gen.load_state_dict(checkpoint, strict=True)
    gen.to(device)
    gen.eval()
    return gen

def get_test_image(idx, image_dir):
    images = os.listdir(image_dir)
    if idx < 0 or idx >= len(images):
        raise IndexError("Index out of range")
    return os.path.join(image_dir, images[idx])

if __name__ == "__main__":
    # Paths to pretrained generator checkpoints
    CHECKPOINTS = {
        "Vangogh": "checkpoints/style_vangogh_pretrained/latest_net_G.pth",
        "Monet": "checkpoints/style_monet_pretrained/latest_net_G.pth",
        "Cezanne": "checkpoints/style_cezanne_pretrained/latest_net_G.pth",
        "Ukiyoe": "checkpoints/style_ukiyoe_pretrained/latest_net_G.pth"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generators = {name: load_pretrained_generator(path, device) for name, path in CHECKPOINTS.items()}

    def process_and_show(image_path):
        input_img = Image.open(image_path).convert("RGB").resize((256, 256))
        input_img_tk = ImageTk.PhotoImage(input_img)
        input_label.config(image=input_img_tk)
        input_label.image = input_img_tk

        for style, gen in generators.items():
            output = test_generator_on_image(image_path, gen, device)
            output_pil = transforms.ToPILImage()(output * 0.5 + 0.5)
            output_img_tk = ImageTk.PhotoImage(output_pil)
            output_labels[style].config(image=output_img_tk)
            output_labels[style].image = output_img_tk

    def open_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            process_and_show(file_path)

    root = tk.Tk()
    root.title("Style Transfer Comparison")

    tk.Button(root, text="Open Image", command=open_image).grid(row=0, column=0, columnspan=5)

    tk.Label(root, text="Input").grid(row=1, column=0)
    for idx, style in enumerate(CHECKPOINTS.keys()):
        tk.Label(root, text=style).grid(row=1, column=idx+1)

    input_label = tk.Label(root)
    input_label.grid(row=2, column=0)
    output_labels = {}
    for idx, style in enumerate(CHECKPOINTS.keys()):
        lbl = tk.Label(root)
        lbl.grid(row=2, column=idx+1)
        output_labels[style] = lbl

    root.mainloop()