from huggingface_hub import hf_hub_download
access_token = "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"

ckpt_path = hf_hub_download(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4-full-ema.ckpt", use_auth_token=access_token)
print(ckpt_path)

# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=access_token)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")

# import torchvision
# from ldm.data.custom import CustomPokemonTrain

# m = CustomPokemonTrain("lambdalabs/pokemon-blip-captions")
# img = m[0]
# from skimage.io import imsave
# imsave('./logs/pokemon.png', img['image'].numpy())