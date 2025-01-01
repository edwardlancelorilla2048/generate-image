import torch
from diffusers import StableDiffusionPipeline

def generate_image(prompt, output_file):
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Prefer GPU if available

    # Remove torch_dtype if running on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    image = pipe(prompt).images[0]
    image.save(output_file)
    print(f"Image saved to {output_file}")

if __name__ == "__main__":
    prompt = "A photorealistic image of a modern AI influencer"
    generate_image(prompt, "ai_influencer.png")
