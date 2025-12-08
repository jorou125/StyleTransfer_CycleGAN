import torch
from models.generator import Generator
from torchvision import transforms
from PIL import Image
import os



test_path = {
    "cezanne": "data/cezanne2photo/cezanne2photo/testB",
    "ukiyoe": "data/ukiyoe2photo/ukiyoe2photo/testB",
    "monet": "data/monet2photo/monet2photo/testB",
    "vangogh": "data/vangogh2photo/vangogh2photo/testB"
}

gen_models_paths = {
    "monet": "checkpoints/style_monet_pretrained/latest_net_G.pth",
    "cezanne": "checkpoints/style_cezanne_pretrained/latest_net_G.pth",
    "ukiyoe": "checkpoints/style_ukiyoe_pretrained/latest_net_G.pth",
    "vangogh": "checkpoints/style_vangogh_pretrained/latest_net_G.pth"
}

output_dir = "outputs/"

def test_generator_on_image(image_path, generator, device, iter=1):
    input_image = Image.open(image_path).convert("RGB").resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        gen = input_tensor
        for _ in range(iter):
            gen = generator(gen)
    return gen.squeeze(0).cpu()

def load_pretrained_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen = Generator()
    gen.load_state_dict(checkpoint, strict=True)
    gen.to(device)
    gen.eval()
    return gen

if __name__ == "__main__":
    iter = 2  # Number of iterations for effective style transfer (only one iter leads to meh results)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for style_name, gen_path in gen_models_paths.items():
        print(f"Processing style: {style_name}")
        generator = load_pretrained_generator(gen_path, device)

        test_images_dir = test_path[style_name]
        for img_name in os.listdir(test_images_dir):
            img_path = os.path.join(test_images_dir, img_name)
            output_tensor = test_generator_on_image(img_path, generator, device, iter=iter)

            output_image = output_tensor.cpu().squeeze(0)
            output_image = (output_image + 1) / 2  # Denormalize
            output_image_pil = transforms.ToPILImage()(output_image)   
            if not os.path.exists(os.path.join(output_dir, f"{style_name}")):
                os.makedirs(os.path.join(output_dir, f"{style_name}"))
            output_image_pil.save(os.path.join(output_dir, f"{style_name}", f"{img_name}"))