import anything2image.imagebind as ib
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
).to(device)
model = ib.imagebind_huge(pretrained=True).eval().to(device)

# generate image
with torch.no_grad():
    thermal_paths =['assets/thermal/030419.jpg']
    embeddings = model.forward({
        ib.ModalityType.THERMAL: ib.load_and_transform_thermal_data(thermal_paths, device),
    }, normalize=True)
    embeddings = embeddings[ib.ModalityType.THERMAL]
    images = pipe(image_embeds=embeddings.half()).images
    images[0].save("thermal2img.png")