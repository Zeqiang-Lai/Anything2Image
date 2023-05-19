import os
import soundfile as sf
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from PIL import Image

from . import imagebind


class Anything2Image:
    def __init__(
        self,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        imagebind_download_dir="checkpoints",
    ):
        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=None if device == 'cpu' else torch.float16,
        ).to(device)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        self.schedulers = {s.__name__: s for s in self.pipe.scheduler.compatibles}
        self.model = imagebind.imagebind_huge(pretrained=True, download_dir=imagebind_download_dir).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self,
                 prompt=None, audio=None, image=None, text=None, depth=None, thermal=None,
                 audio_strength=0.5,
                 noise_level=0, num_inference_steps=20, scheduler='PNDMScheduler',
                 width=768, height=768):
        device, model, pipe = self.device, self.model, self.pipe
        if scheduler is not None:
            pipe.scheduler = self.schedulers[scheduler].from_config(pipe.scheduler.config)
        noise_level = int((self.pipe.image_noising_scheduler.config.num_train_timesteps - 1) * noise_level)

        if audio is not None:
            sr, waveform = audio
            sf.write('tmp.wav', waveform, sr)
            embeddings = model.forward({
                imagebind.ModalityType.AUDIO: imagebind.load_and_transform_audio_data(['tmp.wav'], device),
            })
            audio_embeddings = embeddings[imagebind.ModalityType.AUDIO]
            os.remove('tmp.wav')
        if image is not None:
            Image.fromarray(image).save('tmp.png')
            embeddings = model.forward({
                imagebind.ModalityType.VISION: imagebind.load_and_transform_vision_data(['tmp.png'], device),
            }, normalize=False)
            image_embeddings = embeddings[imagebind.ModalityType.VISION]
            os.remove('tmp.png')

        if depth is not None:
            Image.fromarray(depth).save('tmp.png')
            embeddings = model.forward({
                imagebind.ModalityType.DEPTH: imagebind.load_and_transform_depth_data(['tmp.png'], device),
            }, normalize=True)
            depth_embeddings = embeddings[imagebind.ModalityType.DEPTH]
            os.remove('tmp.png')

        if thermal is not None:
            Image.fromarray(thermal).save('tmp.png')
            embeddings = model.forward({
                imagebind.ModalityType.THERMAL: imagebind.load_and_transform_thermal_data(['tmp.png'], device),
            }, normalize=True)
            thermal_embeddings = embeddings[imagebind.ModalityType.THERMAL]
            os.remove('tmp.png')

        if text is not None and text != "":
            embeddings = self.model.forward({
                imagebind.ModalityType.TEXT: imagebind.load_and_transform_text([text], device),
            }, normalize=False)
            text_embeddings = embeddings[imagebind.ModalityType.TEXT]

        if audio is not None and image is not None:
            embeddings = audio_embeddings * audio_strength + image_embeddings * (1 - audio_strength)
        elif audio is not None and text is not None:
            embeddings = audio_embeddings * audio_strength + text_embeddings * (1 - audio_strength)
        elif image is not None:
            embeddings = image_embeddings
        elif audio is not None:
            embeddings = audio_embeddings
        elif depth is not None:
            embeddings = depth_embeddings
        elif thermal is not None:
            embeddings = thermal_embeddings
        elif text is not None:
            embeddings = text_embeddings
        else:
            embeddings = None

        if embeddings is not None and self.device != 'cpu':
            embeddings = embeddings.half()

        images = pipe(prompt=prompt, image_embeds=embeddings, noise_level=noise_level, num_inference_steps=num_inference_steps, width=width, height=height).images
        return images[0]
