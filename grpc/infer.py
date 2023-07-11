from diffusers import StableDiffusionPipeline
import torch
import uuid

model_id = input("Model location? ")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = input("Prompt? ")
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

filename = 'generated-' + str(uuid.uuid4()) + '.png'
image.save(filename)
print("Saved to: " + filename)
