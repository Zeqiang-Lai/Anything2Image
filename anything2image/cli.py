import os
import fire
from anything2image.api import Anything2Image
import soundfile as sf
from PIL import Image
import numpy as np

def main(
    prompt='', audio=None, image=None, text=None, thermal=None,
    ckpt_dir=os.path.join(os.path.expanduser('~'), 'anything2image', 'checkpoints')
):
    anything2img = Anything2Image(imagebind_download_dir=ckpt_dir)
    if audio is not None: 
        data, samplerate = sf.read(audio)
        audio = (samplerate, data)
    if image is not None: 
        image = np.array(Image.open(image))
    image = anything2img(prompt=prompt, audio=audio, image=image, text=text, thermal=thermal)
    image.save('cli_output.png')

fire.Fire(main)