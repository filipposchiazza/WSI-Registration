from setuptools import setup, find_packages

setup(
    name='WSI-Registration',
    version='0.1.0',
    url='https://github.com/filipposchiazza/WSI-Registration',
    author='Filippo Schiazza',
    description='Algorithm for the registration of whole slide images',
    packages=find_packages(),    
    install_requires=[
        'openslide-python==1.3.1',
        'opencv-python-headless==4.8.1.78',
        'numpy==1.23.5',
        'matplotlib==3.7.1',
        'tqdm==4.65.0',
        'scikit-learn==1.3.2',
        'histomicstk==1.3.0'
        ],
)