# Fast Fourier Transform for Image Processing (Python 3.12)

## Table of Contents
- [Contributors](#contributors)
  
- [Description](#description)

- [How to Run](#how-to-run)


## Contributors

Created by [Marc-Antoine Nadeau](https://github.com/ma-nadeau) and [Karl Bridi](https://github.com/Kalamar136)

## Description

This Python program implements the Fast Fourier Transform.
It was developed and tested using Python 3.12.
## How to Run

To invoke the application, navigate to the `src` directory and use the following command in your terminal:

```bash
cd src
python fft.py [-m mode] [-i image]
```

## Features

### Mode (optional):
- **1. Fast mode (default):** Convert image to FFT form and display.
- **2. Denoise:** The image is denoised by applying an FFT, truncating high frequencies, and then displayed.
- **3. Compress:** Compress image and plot.
- **4. Plot runtime graphs.**

### Image (optional):
- The filename of the image for the DFT (default: `moonlanding.png`).

**Note:** The files should be placed in the `Figures` folder.


