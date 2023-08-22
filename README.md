# MRIQA

## Requirements:
- numpy==1.23.4
- torch==1.13.1
- torchvision==0.14.1


## Usage

### Data augmentation and Fourier transformation

Use `transform.py` to perform data augmentation and Fourier transform.

### Train representation learning

Run command `python main.py --mode="augmentation" --batch_size=32 --projection_dim=128 --epochs=100` for augmented data, and `--mode="fourier"` for Fourier-transformed data.

### Assessment for representations

Use `assessment.py` to conduct quality and representation assessment.
