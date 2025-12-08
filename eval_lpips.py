import os
import torch
import lpips
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


input_paths = {
    "monet": "data/monet2photo/monet2photo/testB",
    "cezanne": "data/cezanne2photo/cezanne2photo/testB",
    "ukiyoe": "data/ukiyoe2photo/ukiyoe2photo/testB",
    "vangogh": "data/vangogh2photo/vangogh2photo/testB"
}

output_paths = {
    "monet": "outputs/monet",
    "cezanne": "outputs/cezanne",
    "ukiyoe": "outputs/ukiyoe",
    "vangogh": "outputs/vangogh"
}

def evaluate_lpips(input_image, output_image, lpips_model, device):
    input_image = input_image.to(device)
    output_image = output_image.to(device)
    with torch.no_grad():
        lpips_value = lpips_model(input_image, output_image)
    return lpips_value.item()

def load_input_output_images(input_image_path, output_image_path, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_image = Image.open(input_image_path).convert("RGB")
    output_image = Image.open(output_image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    output_tensor = transform(output_image).unsqueeze(0).to(device)
    return input_tensor, output_tensor

def print_results(results):
    print("-" * 20)
    print("LPIPS Evaluation Results:")
    for style_name, res in results.items():
        print(f"Style: {style_name}, Average LPIPS: {res['average_lpips']:.4f} over {res['num_images']} images")
    print("-" * 20)

if __name__ == "__main__":
    lpips_model = lpips.LPIPS(net='alex')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model.to(device)
    results = {}

    for style_name in input_paths.keys():
        input_dir = input_paths[style_name]
        output_dir = output_paths[style_name]
        lpips_scores = []

        for img_name in tqdm(os.listdir(input_dir), desc=f"Processing {style_name}"):
            input_image_path = os.path.join(input_dir, img_name)
            output_image_path = os.path.join(output_dir, img_name)

            if not os.path.exists(output_image_path):
                print(f"Output image not found for {img_name}, skipping.")
                continue

            input_tensor, output_tensor = load_input_output_images(input_image_path, output_image_path, device)
            lpips_score = evaluate_lpips(input_tensor, output_tensor, lpips_model, device)
            lpips_scores.append(lpips_score)

        average_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else float('inf')
        results[style_name] = {
            "average_lpips": average_lpips,
            "num_images": len(lpips_scores)
        }

    print_results(results)  