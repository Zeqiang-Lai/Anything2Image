# Anything To Image
<a href='https://huggingface.co/spaces/aaronb/Anything2Image'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

Generate image from anything with [ImageBind](https://github.com/facebookresearch/ImageBind)'s unified latent space and [stable-diffusion-2-1-unclip](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip). 

- No training is need.
- Integration with 🤗  [Diffusers](https://github.com/huggingface/diffusers).
- Online gradio demo with [Huggingface Space](https://huggingface.co/spaces/aaronb/Anything2Image).


Support Tasks

- [Audio to Image](#audio-to-image)
- [Audio+Text to Image](#audiotext-to-image)
- [Audio+Image to Image](#audioimage-to-image)
- [Image to Image](#image-to-image)
- [Text to Image](#text-to-image)

## Audio to Image

| [bird_audio.wav](assets/wav/bird_audio.wav) | [dog_audio.wav](assets/wav/dog_audio.wav) |  [cattle.wav](assets/wav/cattle.wav) | [cat.wav](assets/wav/cat.wav) | 
| --- | --- | --- | --- | 
| ![](assets/generated/audio_to_image/bird_audio.png) | ![](assets/generated/audio_to_image/dog_audio.png) |![](assets/generated/audio_to_image/cattle.png) |![](assets/generated/audio_to_image/cat.png) |

| [fire_engine.wav](assets/wav/fire_engine.wav) | [train.wav](assets/wav/train.wav) |  [motorcycle.wav](assets/wav/motorcycle.wav) | [plane.wav](assets/wav/plane.wav) | 
| --- | --- | --- | --- | 
| ![](assets/generated/audio_to_image/fire_engine.png) | ![](assets/generated/audio_to_image/train.png) |![](assets/generated/audio_to_image/motorcycle.png) |![](assets/generated/audio_to_image/plane.png) |


See [audio2img.py](audio2img.py).

```python
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
with torch.no_grad():
    audio_paths=["assets/wav/bird_audio.wav"]
    embeddings = model.forward({
        imagebind.ModalityType.AUDIO: imagebind.load_and_transform_audio_data(audio_paths, device),
    })
    embeddings = embeddings[imagebind.ModalityType.AUDIO]
    images = pipe(prompt='a painting', image_embeds=embeddings.half()).images
    images[0].save("bird_audio.png")
```

## Audio+Image to Image

Stay tuned

| Image | Audio 1 | Output 1 |  Audio 2  | Output 2 | 
| --- | --- | --- | --- | --- | 
| ![](assets/image/bird.png) | [wave.wav](assets/wav/wave.wav) | ![](assets/generated/audio_image_to_image/bird_wave.png) |  [rain.wav](assets/wav/wave.wav) | ![](assets/generated/audio_image_to_image/bird_rain.png) | 

```python
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
    images = pipe(image_embeds=embeddings.half()).images
    images[0].save("out.png")
```


## Image to Image

| ![](assets/image/dog_image.jpg) | ![](assets/image/bird_image.jpg) |  ![](assets/image/car_image.jpg) | ![](assets/image/room.png) | 
| --- | --- | --- | --- | 
| ![](assets/generated/image_to_image/dog_image.png) | ![](assets/generated/image_to_image/bird_image.png) |![](assets/generated/image_to_image/car_image.png) |![](assets/generated/image_to_image/room.png) |

Top: Input Images. Bottom: Generated Images. See [img2img.py](img2img.py). 

> It is important to set `normalize=False`.

```python
with torch.no_grad():
    paths=["assets/image/dog_image.jpg"]
    embeddings = model.forward({
        imagebind.ModalityType.VISION: imagebind.load_and_transform_vision_data(paths, device),
    }, normalize=False)
    embeddings = embeddings[imagebind.ModalityType.VISION]
    images = pipe(image_embeds=embeddings).images
    images[0].save("out.png")
```

## Text to Image

| A photo of a car. | A sunset over the ocean. | A bird's-eye view of a cityscape.  | A close-up of a flower. | 
| --- | --- | --- | --- | 
| ![](assets/generated/text_to_image/dog_image.png) | ![](assets/generated/text_to_image/bird_image.png) |![](assets/generated/text_to_image/car_image.png) |![](assets/generated/text_to_image/room.png) |

It is not necessary to use ImageBind for text to image. Nervertheless, we show the alignment of ImageBind's text latent space and its image spaces.

```python
with torch.no_grad():
    embeddings = model.forward({
        imagebind.ModalityType.TEXT: imagebind.load_and_transform_text(['a photo of a bird.'], device),
    })
    embeddings = embeddings[imagebind.ModalityType.TEXT]
    images = pipe(image_embeds=embeddings).images
    images[0].save("bird.png")
```

<!-- ## Discussion

Failure cases

| Audio to Image | Audio to Image | Image to Image | 
| --- | --- | --- | 
| [car_audio.wav](assets/wav/car_audio.wav) | [goat.wav](assets/wav/goat.wav) | ![](assets/image/car_image.jpg) | 
| ![](assets/generated/audio_to_image/car_audio.png) | ![](assets/generated/audio_to_image/goat.png)  | ![](assets/generated/image_to_image/car_image.png) |  -->


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
