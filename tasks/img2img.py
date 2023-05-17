import anything2image.imagebind as ib
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
)
pipe = pipe.to(device)

model = ib.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# generate image
with torch.no_grad():
    paths=["assets/image/room.png"]
    embeddings = model.forward({
        ib.ModalityType.VISION: ib.load_and_transform_vision_data(paths, device),
    }, normalize=False)
    embeddings = embeddings[ib.ModalityType.VISION]
    images = pipe(image_embeds=embeddings.half()).images
    images[0].save("img2img.png")
    
