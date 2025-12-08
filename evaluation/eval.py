import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.generator import Generator
import lpips
from torchvision import transforms
from PIL import Image




eval_paths = {
    "eval1": "evaluation/test_images/eval1.jpg",
    "eval2": "evaluation/test_images/eval2.jpg",
    "eval3": "evaluation/test_images/eval3.jpg"
}

style_domains_paths = {
    "monet": "evaluation/style_domains/style_monet",
    "cezanne": "evaluation/style_domains/style_cezanne",
    "ukiyoe": "evaluation/style_domains/style_ukiyoe",
    "vangogh": "evaluation/style_domains/style_vangogh"
}

gen_models_paths = {
    "monet": "checkpoints/style_monet_pretrained/latest_net_G.pth",
    "cezanne": "checkpoints/style_cezanne_pretrained/latest_net_G.pth",
    "ukiyoe": "checkpoints/style_ukiyoe_pretrained/latest_net_G.pth",
    "vangogh": "checkpoints/style_vangogh_pretrained/latest_net_G.pth"
}

output_dir = "evaluation/outputs/"

def load_pretrained_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen = Generator()
    gen.load_state_dict(checkpoint, strict=True)
    gen.to(device)
    gen.eval()
    return gen

def evaluate_content_lpips(input_image, output_image, lpips_model, device):
    input_image = input_image.to(device)
    output_image = output_image.to(device)
    with torch.no_grad():
        lpips_value = lpips_model(input_image, output_image)
    return lpips_value.item()

def evaluate_style_lpips(style_domain_images, output_image, lpips_model, device):
    output_image = output_image.to(device)
    total_lpips = 0.0
    count = 0
    with torch.no_grad():
        for style_image in style_domain_images:
            style_image = style_image.to(device)
            lpips_value = lpips_model(style_image, output_image)
            total_lpips += lpips_value.item()
            count += 1
    average_lpips = total_lpips / count if count > 0 else float('inf')
    return average_lpips

def save_results_json(results, output_path):
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    iter = 2  # Number of iterations for style transfer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    results = {}

    for style_name, gen_path in gen_models_paths.items():
        print(f"Evaluating style: {style_name}")
        generator = load_pretrained_generator(gen_path, device)

        style_domain_images = []
        style_domain_path = style_domains_paths[style_name]
        for img_name in os.listdir(style_domain_path):
            img_path = os.path.join(style_domain_path, img_name)
            img = Image.open(img_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img_tensor = transform(img).unsqueeze(0)
            style_domain_images.append(img_tensor)

        for eval_name, eval_path in eval_paths.items():
            print(f"  Evaluating image: {eval_name}")
            input_img = Image.open(eval_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            input_tensor = transform(input_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output_tensor = input_tensor
                for _ in range(iter):
                    output_tensor = generator(output_tensor)

            content_lpips = evaluate_content_lpips(input_tensor, output_tensor, lpips_model, device)
            style_lpips = evaluate_style_lpips(style_domain_images, output_tensor, lpips_model, device)

            results[f"{style_name}_{eval_name}"] = {
                "content_lpips": content_lpips,
                "style_lpips": style_lpips
            }
            print(f"    Content LPIPS: {content_lpips:.4f}, Style LPIPS: {style_lpips:.4f}")

            output_image = output_tensor.cpu().squeeze(0)
            output_image = (output_image + 1) / 2  # Denormalize
            output_image_pil = transforms.ToPILImage()(output_image)    
            output_image_pil.save(os.path.join(output_dir, f"{style_name}_{eval_name}.png"))
    save_results_json(results, os.path.join(output_dir, "evaluation_results.json"))
    print("Evaluation completed. Results saved.")