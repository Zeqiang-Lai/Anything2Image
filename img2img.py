import imagebind
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
pipe = pipe.to(device)

model = imagebind.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# generate image
with torch.no_grad():
    paths=["assets/image/dog_image.jpg"]
    embeddings = model.forward({
        imagebind.ModalityType.VISION: imagebind.load_and_transform_vision_data(paths, device),
    })
    embeddings = embeddings[imagebind.ModalityType.VISION]
    images = pipe(image_embeds=embeddings.half()).images
    images[0].save("out.png")
    
