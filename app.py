import gradio as gr
import imagebind
import soundfile as sf
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip",
)
pipe = pipe.to(device)

model = imagebind.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

@torch.no_grad()
def anything2img(prompt, audio, image, text):
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
        embeddings = audio_embeddings + image_embeddings
    elif image is not None:
        embeddings = image_embeddings
    elif audio is not None:
        embeddings = audio_embeddings
    else:
        embeddings = None
    
    if text is not None:
        embeddings = model.forward({
            imagebind.ModalityType.TEXT: imagebind.load_and_transform_text([text], device),
        })
        embeddings = embeddings[imagebind.ModalityType.TEXT]
        
    images = pipe(prompt=prompt, image_embeds=embeddings).images
    return images[0]
    

demo = gr.Interface(fn=anything2img, inputs=["text", "audio", "image", "text"], outputs="image")
# demo.launch(server_name='0.0.0.0', server_port=10051, share=True)
demo.launch(server_name='0.0.0.0', server_port=10047, share=True)