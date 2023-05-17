import gradio as gr
import fire
import os
from anything2image.api import Anything2Image


def main(ckpt_dir=os.path.join(os.path.expanduser('~'), 'anything2image', 'checkpoints'), ip='0.0.0.0', port=10049, share=False):
    anything2img = Anything2Image(imagebind_download_dir=ckpt_dir)
    
    with gr.Blocks() as demo:
        gr.HTML(
                """
                <div align='center'> <h1>Anything To Image </h1> </div>
                <p align="center"> Generate image from anything with ImageBind's unified latent space and stable-diffusion-2-1-unclip. </p>
                <p align="center"><a href="https://github.com/Zeqiang-Lai/Anything2Image"><b>https://github.com/Zeqiang-Lai/Anything2Image</b></p>
                """)
        gr.Interface(fn=anything2img, 
                     inputs=["text",
                             "audio", 
                             "image", 
                             "text",
                             ], 
                     outputs="text",
                     examples=[['', 'assets/wav/dog_audio.wav', None, None],
                               ['A painting', 'assets/wav/cat.wav', None, None],
                               ['', 'assets/wav/wave.wav', 'assets/image/bird.png', None],
                               ['', None, 'assets/image/bird_image.jpg', None],
                               ['', None, None, 'A sunset over the ocean.'],
                               ],
                     cache_examples=True,
                     )
    demo.queue(1).launch(server_name=ip, server_port=port, share=share)

fire.Fire(main)