from setuptools import setup, find_packages


setup(
    name='anything2image',
    version='1.0.0',
    packages=find_packages(),
    package_data={
        'anything2image': ['imagebind/bpe/bpe_simple_vocab_16e6.txt.gz']
    },
    include_package_data=True,
    install_requires=[
        'diffusers',
        'pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d',
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
    ],
)
