# Time of Flight Detector Denoising and Reconstruction
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

A deep-learning based online denoising and reconstruction method for time of flight (TOF) detectors in PyTorch.

## Examples
### A randomly generated spectrogram with simulated detector image
![Randomly generated spectrogram with simulated detector image](images/spectrogram_detector_image_3_12.png)

### Denoising and reconstruction of sample with two disabled TOF detectors
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

# Citation

If you find this useful in your research, please consider citing:

[Reconstructing Time-of-Flight Detector Values of Angular Streaking Using Machine Learning.](https://link.aps.org/doi/10.1103/csvm-858f)

    @article{Meier2025,
      title = {Reconstructing time-of-flight detector values of angular streaking using machine learning},
      author = {Meier, David and Viefhaus, Jens and Hartmann, Gregor and Helml, Wolfram and Otto, Thorsten and Sick, Bernhard},
      journal = {Phys. Rev. Accel. Beams},
      volume = {28},
      issue = {7},
      pages = {074601},
      numpages = {21},
      year = {2025},
      month = {6},
      publisher = {American Physical Society},
      doi = {10.1103/csvm-858f},
    }
    
Pacman and real-world dataset is based on:

[Attosecond time–energy structure of X-ray free-electron laser pulses.](https://www.nature.com/articles/s41566-018-0107-6)

    @article{Hartmann2018,
      title = {Attosecond time–energy structure of X-ray free-electron laser pulses},
      volume = {12},
      ISSN = {1749-4893},
      DOI = {10.1038/s41566-018-0107-6},
      number = {4},
      journal = {Nature Photonics},
      publisher = {Springer Science and Business Media LLC},
      author = {Hartmann,  N. and Hartmann,  G. and Heider,  R. and Wagner,  M. S. and Ilchen,  M. and Buck,  J. and Lindahl,  A. O. and Benko,  C. and Gr\"{u}nert,  J. and Krzywinski,  J. and Liu,  J. and Lutman,  A. A. and Marinelli,  A. and Maxwell,  T. and Miahnahri,  A. A. and Moeller,  S. P. and Planas,  M. and Robinson,  J. and Kazansky,  A. K. and Kabachnik,  N. M. and Viefhaus,  J. and Feurer,  T. and Kienberger,  R. and Coffee,  R. N. and Helml,  W.},
      year = {2018},
      month = {3},
      pages = {215–220}
    }
