# pytorch-unet
Pytorch model for unet segmentation (under development)

## Prerequisites

- Linux
- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch 0.4

Installation
------------

    $git clone https://github.com/pedrodiamel/pytorchvision.git
    $cd pytorchvision
    $python setup.py install
    $pip install -r installation.txt

### Visualize result with Visdom

We now support Visdom for real-time loss visualization during training!
To use Visdom in the browser:

    # First install Python server and client 
    pip install visdom
    # Start the server (probably in a screen or tmux)
    python -m visdom.server -env_path runs/visdom/
    # http://localhost:8097/


How use
------------

### Step 1: Train

    ./runs/train.sh
    
### Step 2: Eval
For evaluation we used the notebook [eval.ipynb](./books/test_datacambia.ipynb)

