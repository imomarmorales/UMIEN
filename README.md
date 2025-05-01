# Full Adapted WaterNet

This repository contains the code for the UMIEN, designed for underwater image enhancement. It fuses multiple preprocessed versions of underwater images using a multi-branch CNN architecture optimized for real-time performance on embedded systems.

## Features

- Five input branches: RAW, White Balance, Histogram Equalization, Gamma Correction, and MSRCR.
- Parallel feature extraction using convolutional layers.
- Feature fusion and refinement for enhanced RGB image reconstruction.
- Lightweight model suitable for deployment on Raspberry Pi or Jetson Nano.
- Designed for underwater images with color distortion, low contrast, and turbidity.

## Files

- `trainR.py`: Updated training script (20 epochs, batch size 16, GPU-ready).
- `UMIEN.py`: CNN architecture definition.
- `dataset.py`: Dataset loader with preprocessing steps.
- `inferenciareal.py`: Inference script for evaluating enhanced images.

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- OpenCV
- NumPy
- Pillow

## Training

To train the model:

```bash
python train_revised.py
```

Ensure you have a GPU available or it will fallback to CPU automatically.

## Citation

If you use this code in your research, please cite the corresponding paper.




## License

MIT License