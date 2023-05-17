import gradio as gr
import fire
import os
from anything2image.api import Anything2Image


def main(ckpt_dir=os.path.join(os.path.expanduser('~'), 'anything2image', 'checkpoints'), ip='0.0.0.0', port=10049):
    anything2img = Anything2Image(imagebind_download_dir=ckpt_dir)
    demo = gr.Interface(fn=anything2img, inputs=["text", "audio", "image", "text"], outputs="image")
    demo.queue(1).launch(server_name=ip, server_port=port)

fire.Fire(main)