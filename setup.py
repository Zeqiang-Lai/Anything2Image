from setuptools import setup, find_packages
import os

image_dir = 'anything2image/assets/image'
image_data = [os.path.join('assets/image', name) for name in os.listdir(image_dir)]

wav_dir = 'anything2image/assets/wav'
wav_data = [os.path.join('assets/wav', name) for name in os.listdir(wav_dir)]

depth_dir = 'anything2image/assets/depth'
depth_data = [os.path.join('assets/depth', name) for name in os.listdir(depth_dir)]

thermal_dir = 'anything2image/assets/thermal'
thermal_data = [os.path.join('assets/thermal', name) for name in os.listdir(thermal_dir)]


setup(
    name='anything2image',
    version='1.1.4',
    packages=find_packages(),
    package_data={
        'anything2image': ['imagebind/bpe/bpe_simple_vocab_16e6.txt.gz'] + image_data + wav_data + depth_data + thermal_data
    },
    include_package_data=True,
    install_requires=[
        'diffusers',
        'timm==0.6.7',
        'ftfy',
        'regex',
        'einops',
        'fvcore',
        'decord==0.6.0',
        'soundfile',
        'transformers',
        'gradio',
        'fire',
        'pytorchvideo',
        'accelerate'
    ],
)
