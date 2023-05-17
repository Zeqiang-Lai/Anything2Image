from setuptools import setup, find_packages


setup(
    name='anything2image',
    version='1.0.5',
    packages=find_packages(),
    package_data={
        'anything2image': ['imagebind/bpe/bpe_simple_vocab_16e6.txt.gz']
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
