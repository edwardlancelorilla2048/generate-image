from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt, output_file):
    # Load Stable Diffusion model from Hugging Face
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the image
    image = pipe(prompt).images[0]

    # Save the image to a file
    image.save(output_file)
    print(f"Image saved to {output_file}")

if __name__ == "__main__":
    prompt = "A photorealistic AI influencer in a futuristic setting, wearing stylish clothing, vibrant neon lights"
    generate_image(prompt, "ai_influencer.png")
