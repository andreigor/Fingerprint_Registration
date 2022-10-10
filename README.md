
# Latent Fingerprint Registration

This is a module of fingerprint registration, which is part of a MSc. project
developed in the Computing Institute of Unicamp, in partership with Griaule.

This repository explores a few techniques for fingerprint image registration, specifically finding homologous points, as 
described in chapter 5 of the book Theory and Applications of Image Registration - Goshtasby 2017.

Thus far, the following techniques are implemented:


Several Dithering masks have been proposed by the literature. In this project, we implemented the following:
- a) Parameter estimation by clustering
- b) Parameter estimation by RANSAC

The feature points are the fingerprints minutias, extracted directly from the pre-trained network [Fingernet](https://github.com/592692070/FingerNet) 


## How to use

1. run ```cd``` in shell into directory ```src```
2. run ```python input_image.png dithering_mask output_image.png```

##

The output image will be saved in the ```outputs``` directory. A few sample images are availabel in the ```images``` directory


## Authors

- [@andrenobrega](https://github.com/andreigor)


## Acknowledgements

 - [Griaule](https://griaule.com/)
 - [Computer Institute - Unicamp](https://ic.unicamp.br/)


## Run Locally

Clone the project

```bash
  https://github.com/andreigor/Fingerprint_Registration
```

Go to the project directory

```bash
  cd fingerprint_registration
```

Install dependencies

```bash
  pip install requirements.txt
```

Go to source directory

```bash
  cd src
```

Run sample example

```bash
python3 fingerprint_registration.py ../input_samples/sample_comparisons.txt ../input_samples/sample_images/ ../input_samples/sample_mnts/ sample_run
```