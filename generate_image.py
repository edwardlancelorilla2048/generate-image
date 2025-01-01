from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt, output_file):
    # Load photorealistic model
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"  # Photorealistic Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the image
    image = pipe(prompt).images[0]

    # Save the image to a file
    image.save(output_file)
    print(f"Image saved to {output_file}")

if __name__ == "__main__":
    prompt = "A photorealistic image of a modern AI influencer, stylishly dressed, in a futuristic setting"
    generate_image(prompt, "ai_influencer.png")
