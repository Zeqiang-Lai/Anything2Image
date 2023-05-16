# Anything To Image
<img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/spaces/aaronb/Anything2Image'>

Generate image from anything with [ImageBind](https://github.com/facebookresearch/ImageBind)'s unified latent space and [stable-diffusion-2-1-unclip](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip). 

- No training is need.
- Integration with 🤗  [Diffusers](https://github.com/huggingface/diffusers).
- `imagebind` is directly copy from [official repo](https://github.com/facebookresearch/ImageBind) without any modification. 
- Online gradio demo with [Huggingface Space](https://huggingface.co/spaces/aaronb/Anything2Image).


Support Tasks

- [Audio to Image](#audio-to-image)
- [Audio+Text to Image](#audiotext-to-image)

## Audio to Image

| [bird_audio.wav](assets/wav/bird_audio.wav) | [dog_audio.wav](assets/wav/dog_audio.wav) |  [cattle.wav](assets/wav/cattle.wav) | [cat.wav](assets/wav/cat.wav) | 
| --- | --- | --- | --- | 
| ![](assets/generated/audio_to_image/bird_audio.png) | ![](assets/generated/audio_to_image/dog_audio.png) |![](assets/generated/audio_to_image/cattle.png) |![](assets/generated/audio_to_image/cat.png) |

See [audio2img.py](audio2img.py).

```python
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
    audio_paths=["assets/wav/bird_audio.wav"]
    embeddings = model.forward({
        imagebind.ModalityType.AUDIO: imagebind.load_and_transform_audio_data(audio_paths, device),
    })
    embeddings = embeddings[imagebind.ModalityType.AUDIO]
    images = pipe(image_embeds=embeddings.half()).images
    images[0].save("bird_audio.png")
```

## Audio+Text to Image 


| [cat.wav](assets/wav/cat.wav) | [cat.wav](assets/wav/cat.wav) |  [bird_audio.wav](assets/wav/bird_audio.wav) | [bird_audio.wav](assets/wav/bird_audio.wav) | 
| --- | --- | --- | --- | 
| A painting    | A photo    |  A painting   |  A photo   | 
| ![](assets/generated/audio_text_to_image/cat_a_painting.png) | ![](assets/generated/audio_text_to_image/cat_a_photo.png) |![](assets/generated/audio_text_to_image/bird_a_painting.png) |![](assets/generated/audio_text_to_image/bird_a_photo.png) |


See [audiotext2img.py](audiotext2img.py).

```python
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
    audio_paths=["assets/wav/bird_audio.wav"]
    embeddings = model.forward({
        imagebind.ModalityType.AUDIO: imagebind.load_and_transform_audio_data(audio_paths, device),
    })
    embeddings = embeddings[imagebind.ModalityType.AUDIO]
    images = pipe(prompt='a painting', image_embeds=embeddings.half()).images
    images[0].save("bird_audio.png")
```

## Citation

Latent Diffusion

```bibtex
@InProceedings{Rombach_2022_CVPR,
    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10684-10695}
}
```

ImageBind
```bibtex
@inproceedings{girdhar2023imagebind,
  title={ImageBind: One Embedding Space To Bind Them All},
  author={Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuang
and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2023}
}
```
