import subprocess
import sys

def run_pip(package):
    print(f"Installing {package}...")
    subprocess.run([r"d:\voice-cloner\env311\Scripts\python.exe", "-m", "pip", "install", package], check=False)

deps = [
    "setuptools==81.0.0",
    "numpy==2.3.5",
    "librosa==0.11.0",
    "scipy==1.16.3",
    "faiss-cpu==1.13.2",
    "torchcrepe",
    "torchfcpe",
    "tqdm",
    "rich",
    "requests<2.32.0",
    "transformers==4.44.2",
    "tensorboard",
    "tensorboardX",
    "omegaconf",
    "jinja2",
    "MarkupSafe",
    "fsspec",
    "pillow",
    "soundfile",
    "noisereduce",
    "pedalboard",
    "stftpitchshift",
    "soxr",
    "matplotlib",
    "gradio==6.9.0"
]

for d in deps:
    run_pip(d)
