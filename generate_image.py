import torch
from diffusers import StableDiffusionPipeline

def generate_true_photorealistic_image(prompt, output_file):
    # Use a fine-tuned model specialized for photorealism
    model_id = "realistic-vision-v2"  # Replace with a truly photorealistic model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use appropriate dtype for the device
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Generate the image
    image = pipe(prompt).images[0]
    image.save(output_file)
    print(f"Photorealistic image saved to {output_file}")

if __name__ == "__main__":
    prompt = (
        "A photorealistic image of a young influencer with natural features, stylish modern outfit, "
        "sitting outdoors, realistic skin texture, professional camera quality, bright daylight"
    )
    generate_true_photorealistic_image(prompt, "ai_influencer_realistic.png")
