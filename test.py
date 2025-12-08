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

    image_dir = None
    images = []

    def select_folder():
        global image_dir, images
        image_dir = filedialog.askdirectory(title="Select Image Folder")
        if image_dir:
            images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            idx_entry.config(state="normal")
            idx_entry.delete(0, tk.END)
            idx_entry.insert(0, "0")
            idx_entry.config(state="normal")
            process_and_show(0)

    def process_and_show(idx):
        if not images or idx < 0 or idx >= len(images):
            return
        image_path = os.path.join(image_dir, images[idx])
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

    def on_index_enter(event=None):
        try:
            idx = int(idx_entry.get())
            process_and_show(idx)
        except Exception:
            pass

    root = tk.Tk()
    root.title("Style Transfer Comparison")

    tk.Button(root, text="Select Image Folder", command=select_folder).grid(row=0, column=0, columnspan=5)

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

    tk.Label(root, text="Image Index:").grid(row=3, column=0)
    idx_entry = tk.Entry(root, width=5, state="disabled")
    idx_entry.grid(row=3, column=1)
    idx_entry.bind("<Return>", on_index_enter)
    tk.Button(root, text="Show", command=on_index_enter).grid(row=3, column=2)

    root.mainloop()