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
    audio_paths=["assets/bird_audio.wav"]
    embeddings = model.forward({
        imagebind.ModalityType.AUDIO: imagebind.load_and_transform_audio_data(audio_paths, device),
    })
    embeddings = embeddings[imagebind.ModalityType.AUDIO]
    images = pipe(image_embeds=embeddings.half()).images
    images[0].save("bird_audio.png")