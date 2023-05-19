import gradio as gr
import fire
import os
from anything2image.api import Anything2Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def clear(*args):
    return [None for _ in args]


def main(ckpt_dir=os.path.join(os.path.expanduser('~'), 'anything2image', 'checkpoints'), ip=None, port=None, share=False):
    anything2img = Anything2Image(imagebind_download_dir=ckpt_dir)

    with gr.Blocks() as demo:
        gr.HTML(
            """
            <div align='center'> <h1>Anything To Image </h1> </div>
            <p align="center"> Generate image from anything with ImageBind's unified latent space and stable-diffusion-2-1-unclip. </p>
            <p align="center"><a href="https://github.com/Zeqiang-Lai/Anything2Image"><b>https://github.com/Zeqiang-Lai/Anything2Image</b></p>
            """,
        )
        with gr.Accordion("Options", open=False):
            with gr.Row():
                noise_level = gr.Slider(0, 1, value=0, label="Noise Level",
                                        info="The amount of noise to add to the image embeddings. A higher `noise_level` increases the variance in the final un-noised images.")
                num_inference_steps = gr.Slider(5, 100, value=20, step=1, label="Num Inference Steps",
                                                info='The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference')
                scheduler = gr.Dropdown(choices=list(anything2img.schedulers.keys()), value='PNDMScheduler',
                                        label='Scheduler', info='A scheduler to be used in combination with `unet` to denoise the encoded image latents.')
                with gr.Column():
                    gr.Markdown('The default image size is 768 x 768, other size might generate inferior results.')
                    height = gr.Slider(0, 1280, value=768, step=8, label='Image Height', info='Too larger value might cause OOM error.')
                    width = gr.Slider(0, 1280, value=768, step=8, label='Image Width', info='Too larger value might cause OOM error.')

        with gr.Tab('Audio to Image'):
            wav_dir = os.path.join(CURRENT_DIR, 'assets/wav')

            def audio2image(audio, noise_level, num_inference_steps, scheduler, height, width):
                return anything2img(audio=audio, noise_level=noise_level, num_inference_steps=num_inference_steps, scheduler=scheduler,
                                    height=height, width=width)
            with gr.Row():
                with gr.Column():
                    audio = gr.Audio()
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", variant='primary')
                with gr.Column():
                    output = gr.Image()
                submit_btn.click(audio2image, inputs=[audio, noise_level, num_inference_steps, scheduler, height, width], outputs=[output])
                clear_btn.click(fn=clear, inputs=[audio, output], outputs=[audio, output])
            with gr.Row():
                gr.Examples([os.path.join(wav_dir, name) for name in os.listdir(wav_dir)], inputs=[audio])

        with gr.Tab('Audio+Text to Image'):
            wav_dir = os.path.join(CURRENT_DIR, 'assets/wav')

            def audiotext2image(embed_mode, prompt, audio, audio_strength, noise_level, num_inference_steps, scheduler, height, width):
                if embed_mode:
                    return anything2img(text=prompt, audio=audio, noise_level=noise_level, num_inference_steps=num_inference_steps, scheduler=scheduler,
                                        height=height, width=width, audio_strength=audio_strength)
                return anything2img(prompt=prompt, audio=audio, noise_level=noise_level, num_inference_steps=num_inference_steps, scheduler=scheduler,
                                    height=height, width=width)

            def switch_mode(mode, strength):
                return mode, gr.update(visible=mode)

            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label='Text Prompt (Optional)')
                    audio = gr.Audio()
                    mode = gr.Checkbox(False, label='Embedding Fusion Mode',
                                       info='Whether to use ImageBind text embedding for fusion.')
                    audio_strength = gr.Slider(0, 1, value=0.5, label="Audio Strength",
                                               info="A higher audio strength makes the output image align more audio",
                                               visible=False)
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", variant='primary')
                with gr.Column():
                    output = gr.Image()
                mode.change(switch_mode, [mode, audio_strength], [mode, audio_strength])
                submit_btn.click(audiotext2image,
                                 inputs=[mode, prompt, audio, audio_strength, noise_level, num_inference_steps, scheduler, height, width],
                                 outputs=[output])
                clear_btn.click(fn=clear, inputs=[audio, prompt, output], outputs=[audio, prompt, output])
            with gr.Row():
                gr.Examples(['A painting', 'A photo'], inputs=[prompt], label='Example Prompts')
            with gr.Row():
                gr.Examples([os.path.join(wav_dir, name) for name in os.listdir(wav_dir)], inputs=[audio], label='Example Audios')

        with gr.Tab('Audio+Image to Image'):
            image_dir = os.path.join(CURRENT_DIR, 'assets/image')

            def audioimage2image(audio, image, audio_strength, noise_level, num_inference_steps, scheduler, height, width):
                return anything2img(image=image, audio=audio, audio_strength=audio_strength, noise_level=noise_level, num_inference_steps=num_inference_steps, scheduler=scheduler,
                                    height=height, width=width)

            with gr.Row():
                with gr.Column():
                    audio = gr.Audio()
                    image = gr.Image()
                    audio_strength = gr.Slider(0, 1, value=0.5, label="Audio Strength", info="A higher audio strength makes the output image align more audio")
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", variant='primary')
                with gr.Column():
                    output = gr.Image()
                submit_btn.click(audioimage2image, inputs=[audio, image, audio_strength, noise_level, num_inference_steps, scheduler, height, width], outputs=[output])
                clear_btn.click(fn=clear, inputs=[audio, image, output], outputs=[audio, image, output])
            with gr.Row():
                gr.Examples([os.path.join(CURRENT_DIR, 'assets/wav/wave.wav'), os.path.join(CURRENT_DIR, 'assets/wav/rain.wav')],
                            inputs=[audio],
                            label='Example Audios')
            with gr.Row():
                gr.Examples([os.path.join(image_dir, name) for name in os.listdir(image_dir)], inputs=[image], label='Example Images')

        with gr.Tab('Image to Image'):
            image_dir = os.path.join(CURRENT_DIR, 'assets/image')

            def image2image(image, noise_level, num_inference_steps, scheduler, height, width):
                return anything2img(image=image, noise_level=noise_level, num_inference_steps=num_inference_steps, scheduler=scheduler,
                                    height=height, width=width)

            with gr.Row():
                with gr.Column():
                    image = gr.Image()
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", variant='primary')
                with gr.Column():
                    output = gr.Image()
                submit_btn.click(image2image, inputs=[image, noise_level, num_inference_steps, scheduler, height, width], outputs=[output])
                clear_btn.click(fn=clear, inputs=[image, output], outputs=[image, output])
            with gr.Row():
                gr.Examples([os.path.join(image_dir, name) for name in os.listdir(image_dir)], inputs=[image])

        with gr.Tab('Text to Image'):
            def text2image(text, noise_level, num_inference_steps, scheduler, height, width):
                return anything2img(text=text, noise_level=noise_level, num_inference_steps=num_inference_steps, scheduler=scheduler,
                                    height=height, width=width)

            with gr.Row():
                with gr.Column():
                    text = gr.Textbox()
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", variant='primary')
                with gr.Column():
                    output = gr.Image()
                submit_btn.click(text2image, inputs=[text, noise_level, num_inference_steps, scheduler, height, width], outputs=[output])
                clear_btn.click(fn=clear, inputs=[text, output], outputs=[text, output])
            with gr.Row():
                gr.Examples(['A sunset over the ocean.',
                             'A photo of a car',
                             "A bird's-eye view of a cityscape.",
                             "A close-up of a flower."], inputs=[text])

        with gr.Tab('Thermal to Image'):
            thermal_dir = os.path.join(CURRENT_DIR, 'assets/thermal')

            def thermal2image(thermal, noise_level, num_inference_steps, scheduler, height, width):
                return anything2img(thermal=thermal, noise_level=noise_level, num_inference_steps=num_inference_steps, scheduler=scheduler,
                                    height=height, width=width)

            with gr.Row():
                with gr.Column():
                    image = gr.Image()
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", variant='primary')
                with gr.Column():
                    output = gr.Image()
                submit_btn.click(thermal2image, inputs=[image, noise_level, num_inference_steps, scheduler, height, width], outputs=[output])
                clear_btn.click(fn=clear, inputs=[image, output], outputs=[image, output])
            with gr.Row():
                gr.Examples([os.path.join(thermal_dir, name) for name in os.listdir(thermal_dir)], inputs=[image])

        with gr.Tab('Text+Any to Image'):
            def textany2image(prompt, audio, image, audio_strength, noise_level, num_inference_steps, scheduler, height, width):
                return anything2img(prompt=prompt, image=image, audio=audio, audio_strength=audio_strength, noise_level=noise_level, num_inference_steps=num_inference_steps, scheduler=scheduler,
                                    height=height, width=width)

            with gr.Row():
                with gr.Column():
                    text = gr.Textbox()
                    audio = gr.Audio()
                    image = gr.Image()
                    audio_strength = gr.Slider(0, 1, value=0.5, label="Audio Strength", info="A higher audio strength makes the output image align more audio")
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", variant='primary')
                with gr.Column():
                    output = gr.Image()
                submit_btn.click(textany2image, inputs=[text, audio, image, audio_strength, noise_level, num_inference_steps, scheduler, height, width], outputs=[output])
                clear_btn.click(fn=clear, inputs=[text, audio, image, output], outputs=[text, audio, image, output])
            with gr.Row():
                gr.Examples([['A painting.', os.path.join(CURRENT_DIR, 'assets/image/bird.png'), os.path.join(CURRENT_DIR, 'assets/wav/wave.wav')]], inputs=[text, image, audio])

    demo.launch(server_name=ip, server_port=port, share=share, allowed_paths=[os.path.join(CURRENT_DIR, 'assets')])


fire.Fire(main)
