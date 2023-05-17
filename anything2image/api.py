import soundfile as sf
import torch
import numpy as np
from diffusers import StableUnCLIPImg2ImgPipeline
from PIL import Image

from . import imagebind


class Anything2Image:
    def __init__(
        self, 
        device = "cuda:0" if torch.cuda.is_available() else "cpu",
        imagebind_download_dir="checkpoints"
    ):
        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=None if device == 'cpu' else torch.float16,
        ).to(device)
        self.model = imagebind.imagebind_huge(pretrained=True, download_dir=imagebind_download_dir).eval().to(device)
        self.device = device
        
    @torch.no_grad()
    def __call__(self, prompt=None, audio=None, image=None, text=None):
        device, model, pipe = self.device, self.model, self.pipe
        
        if audio is not None:
            sr, waveform = audio
            sf.write('tmp.wav', waveform, sr)
            embeddings = model.forward({
                imagebind.ModalityType.AUDIO: imagebind.load_and_transform_audio_data(['tmp.wav'], device),
            })
            audio_embeddings = embeddings[imagebind.ModalityType.AUDIO]
        if image is not None:
            Image.fromarray(image).save('tmp.png')
            embeddings = model.forward({
                imagebind.ModalityType.VISION: imagebind.load_and_transform_vision_data(['tmp.png'], device),
            }, normalize=False)
            image_embeddings = embeddings[imagebind.ModalityType.VISION]
            
        if audio is not None and image is not None:
            embeddings = (audio_embeddings + image_embeddings) / 2
        elif image is not None:
            embeddings = image_embeddings
        elif audio is not None:
            embeddings = audio_embeddings
        else:
            embeddings = None
        
        if text is not None and text != "":
            embeddings = self.model.forward({
                imagebind.ModalityType.TEXT: imagebind.load_and_transform_text([text], device),
            }, normalize=False)
            embeddings = embeddings[imagebind.ModalityType.TEXT]
        
        if embeddings is not None and self.device != 'cpu':
            embeddings = embeddings.half()
        
        images = pipe(prompt=prompt, image_embeds=embeddings).images
        return images[0]