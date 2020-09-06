### Installation

#### CUDA on WSL2

https://docs.nvidia.com/cuda/wsl-user-guide/index.html


sudo apt-get install -y cuda-toolkit-11-0


#### Fix connect to ipykernel error (vscode)

python -m pip install 'traitlets==4.3.3' --force-reinstall

### Install Torch etc.
```bash
 pipenv install torch transformers torchvision --skip-lock
 pipenv install pytorch_transformers --skip-lock
 pipenv install fastai --skip-lock
```
