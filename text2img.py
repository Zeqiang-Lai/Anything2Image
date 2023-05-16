import imagebind
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
)
pipe = pipe.to(device)

model = imagebind.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# generate image
with torch.no_grad():
    embeddings = model.forward({
        imagebind.ModalityType.TEXT: imagebind.load_and_transform_text(['A photo of a car.'], device),
    }, normalize=False)
    embeddings = embeddings[imagebind.ModalityType.TEXT]
    images = pipe(image_embeds=embeddings).images
    images[0].save("bird.png")