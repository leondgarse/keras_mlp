from setuptools import find_packages
from setuptools import setup

setup(
    name="keras-mlp",
    version="1.0.0",
    author="Leondgarse",
    author_email="leondgarse@google.com",
    url="https://github.com/leondgarse/keras_mlp",
    description="keras mlp_mixer and res_mlp",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "tensorflow",
    ],
    packages=find_packages(),
    license="Apache 2.0",
)
