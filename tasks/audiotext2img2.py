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
    audio_paths = ["assets/wav/dog_audio.wav"]
    embeddings = model.forward({
        ib.ModalityType.AUDIO: ib.load_and_transform_audio_data(audio_paths, device),
    })
    audio_embeddings = embeddings[ib.ModalityType.AUDIO]
    embeddings = model.forward({
        ib.ModalityType.TEXT: ib.load_and_transform_text(['tree'], device),
    }, normalize=False)
    text_embeddings = embeddings[ib.ModalityType.TEXT]
    
    w = 0.5
    embeddings = text_embeddings * w + audio_embeddings * (1-w)
    images = pipe(image_embeds=embeddings.half()).images
    images[0].save("audiotext2img2.png")
