# Time of Flight Detector Denoising and Reconstruction
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

A deep-learning based online denoising and reconstruction method for time of flight (TOF) detectors in PyTorch.

![Randomly generated spectrogram with simulated detector image](images/spectrogram_detector_image.png)

![Denoising and reconstruction of sample with two disabled TOF detectors](images/two_tofs_disabled.png)

# Quick Start

```py
from evaluation import Evaluator

# Initialize evaluator
e: Evaluator = Evaluator({}, output_dir="outputs/", load_max=0)
# Create spectrogram with 3 peaks with seed 12 and simulate detector image
e.plot_spectrogram_detector_image(3, 12)

# Initialize evaluator with General model
model_dict = {
     "General model": "outputs/tof_reconstructor/hj69jsmh/checkpoints"}
e: Evaluator = Evaluator(model_dict, output_dir="outputs/", load_max=10)

# Disable TOF detectors on position #8 and #13
e.plot_reconstructing_tofs_comparison([7, 12], "General model")
```

