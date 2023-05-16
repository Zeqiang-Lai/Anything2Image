import gradio as gr
import imagebind
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
pipe = pipe.to(device)

model = imagebind.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

@torch.no_grad()
def anything2img(prompt, audio):
    sr, waveform = audio
    audio_path = 'tmp.wav'
    sf.write(audio_path, waveform, sr)
    audio_paths=[audio_path]
    embeddings = model.forward({
        imagebind.ModalityType.AUDIO: imagebind.load_and_transform_audio_data(audio_paths, device),
    })
    embeddings = embeddings[imagebind.ModalityType.AUDIO]
    images = pipe(prompt=prompt, image_embeds=embeddings.half()).images
    return images[0]
    

demo = gr.Interface(fn=anything2img, inputs=["text", "audio"], outputs="image")
# demo.launch(server_name='0.0.0.0', server_port=10051, share=True)
demo.launch(server_name='0.0.0.0', server_port=10047, share=True)