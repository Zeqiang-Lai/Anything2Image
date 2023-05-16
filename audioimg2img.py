import imagebind
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip"
)
pipe = pipe.to(device)

model = imagebind.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# generate image
with torch.no_grad():
    embeddings = model.forward({
        imagebind.ModalityType.VISION: imagebind.load_and_transform_vision_data(["assets/image/bird.png"], device),
    })
    img_embeddings = embeddings[imagebind.ModalityType.VISION]
    embeddings = model.forward({
        imagebind.ModalityType.AUDIO: imagebind.load_and_transform_audio_data(["assets/wav/wave.wav"], device),
    }, normalize=False)
    audio_embeddings = embeddings[imagebind.ModalityType.AUDIO]
    embeddings = img_embeddings + audio_embeddings
    images = pipe(image_embeds=embeddings).images
    images[0].save("out.png")
    
