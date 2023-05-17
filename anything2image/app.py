import gradio as gr
import fire
import os
from anything2image.api import Anything2Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(ckpt_dir=os.path.join(os.path.expanduser('~'), 'anything2image', 'checkpoints'), ip=None, port=None, share=False):
    anything2img = Anything2Image(imagebind_download_dir=ckpt_dir)

    with gr.Blocks() as demo:
        gr.HTML(
            """
            <div align='center'> <h1>Anything To Image </h1> </div>
            <p align="center"> Generate image from anything with ImageBind's unified latent space and stable-diffusion-2-1-unclip. </p>
            <p align="center"><a href="https://github.com/Zeqiang-Lai/Anything2Image"><b>https://github.com/Zeqiang-Lai/Anything2Image</b></p>
            """
        )
        with gr.Tab('Audio to Image'):
            wav_dir = os.path.join(CURRENT_DIR, 'assets/wav')
            def audio2image(audio): return anything2img(audio=audio)
            gr.Interface(
                fn=audio2image,
                inputs="audio",
                outputs="image",
                examples=[os.path.join(wav_dir, name) for name in os.listdir(wav_dir)],
            )
        with gr.Tab('Audio+Text to Image'):
            wav_dir = os.path.join(CURRENT_DIR, 'assets/wav')
            def audiotext2image(prompt, audio): return anything2img(prompt=prompt, audio=audio)
            gr.Interface(
                fn=audiotext2image,
                inputs=["text","audio"],
                outputs="image",
                examples=[
                    ['A painting', os.path.join(CURRENT_DIR, 'assets/wav/cat.wav')],
                    ['A photo', os.path.join(CURRENT_DIR, 'assets/wav/cat.wav')],
                    ['A painting', os.path.join(CURRENT_DIR, 'assets/wav/dog_audio.wav')],
                    ['A photo', os.path.join(CURRENT_DIR, 'assets/wav/dog_audio.wav')],
                ],
            )
        with gr.Tab('Audio+Image to Image'):
            wav_dir = 'assets/wav'
            def audioimage2image(audio, image): return anything2img(image=image, audio=audio)
            gr.Interface(
                fn=audioimage2image,
                inputs=["audio","image"],
                outputs="image",
                examples=[
                    [os.path.join(CURRENT_DIR, 'assets/wav/wave.wav'), os.path.join(CURRENT_DIR, 'assets/image/bird.png')],
                    [os.path.join(CURRENT_DIR, 'assets/wav/wave.wav'), os.path.join(CURRENT_DIR, 'assets/image/dog_image.jpg')],
                    [os.path.join(CURRENT_DIR, 'assets/wav/wave.wav'), os.path.join(CURRENT_DIR, 'assets/image/room.png')],
                    [os.path.join(CURRENT_DIR, 'assets/wav/rain.wav'), os.path.join(CURRENT_DIR, 'assets/image/room.png')],
                ],
            )
        with gr.Tab('Image to Image'):
            image_dir = os.path.join(CURRENT_DIR, 'assets/image')
            def image2image(image): return anything2img(image=image)
            gr.Interface(
                fn=image2image,
                inputs=["image"],
                outputs="image",
                examples=[os.path.join(image_dir, name) for name in os.listdir(image_dir)],
            )
        with gr.Tab('Text to Image'):
            def text2image(text): return anything2img(text=text)
            gr.Interface(
                fn=text2image,
                inputs=["text"],
                outputs="image",
                examples=['A sunset over the ocean.', 
                          'A photo of a car', 
                          "A bird's-eye view of a cityscape.", 
                          "A close-up of a flower."],
            )
        with gr.Tab('Text+Any to Image'):
            def textany2image(prompt, image, audio): return anything2img(prompt=prompt, image=image, audio=audio)
            gr.Interface(
                fn=textany2image,
                inputs=["text", "image", "audio"],
                outputs="image",
                examples=[['A painting.', os.path.join(CURRENT_DIR, 'assets/image/bird.png'), os.path.join(CURRENT_DIR, 'assets/wav/wave.wav')]],
            )
        
    demo.queue(1).launch(server_name=ip, server_port=port, share=share)


fire.Fire(main)