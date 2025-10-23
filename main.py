import torch
from diffusers import StableDiffusionPipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
def generate_image(prompt):
    image = pipe(prompt).images[0]
    image.save("output.png")
prompt = input("Enter your image prompt: ")
generate_image(prompt)
