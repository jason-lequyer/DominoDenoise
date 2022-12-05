# Domino Denoise

The most accurate state-of-the-art denoisers typically train on a representative dataset. But gathering training data is not always easy or feasible, so interest has grown in blind zero-shot denoisers that train only on the image they are denoising. The most accurate blind-zero shot methods are blind-spot networks, which mask pixels and attempt to infer them from their surroundings. Other methods exist where all neurons participate in forward inference, however they are not as accurate and are susceptible to overfitting. Here we present a hybrid approach. We first introduce a semi blind-spot network where the network can see only a small percentage of inputs during gradient update. We then resolve overfitting by introducing a validation scheme where we split pixels into two groups and fill in pixel gaps using a domino tiling based validation scheme. Our method achieves an average PSNR increase of 0.28 and a three fold increase in speed over Self2Self on synthetic Gaussian noise.

![alt text](https://github.com/pelletierlab/DominoDenoise/blob/main/5.png)

# Installation
First download our code by clicking Code -> Download ZIP in the top right corner and unzip it on your computer.

If you don't already have anaconda, install it by following instructions at this link: https://docs.anaconda.com/anaconda/install/.

It would also be helpful to have ImageJ installed: https://imagej.nih.gov/ij/download.html.

Open Anaconda Prompt (or terminal if on Mac/Linux) and enter the following commands to create a new conda environment and install the required packages:

```python
conda create --name DD
conda activate DD
conda install -c pytorch pytorch=1.12.0
conda install -c conda-forge tifffile=2021.7.2
conda install -c anaconda scipy=1.7.3
```
If the installs don't work, removing the specific version may fix this. To do this, remove everything after the equals sign, including the equals sign (e.g. conda install -c conda-forge tifffile).
# Using Domino Denoise on your 2D grayscale data

Create a folder in the master directory (the directory that contains DD.py) and put your noisy images into it. Then open anaconda prompt/terminal and run the following:

```python
cd <masterdirectoryname>
conda activate DD
python DD.py <noisyfolder>
```
Replacing "masterdirectoryname" with the full path to the directory that contains DD.py, and replacing "noisyfolder" with the name of the folder containing images you want denoised. Results will be saved to the directory '<noisyolder>_N2F'. Issues may arise if using an image format that is not supported by the tifffile python package, to fix these issues you can open your images in ImageJ and re-save them as .tif (even if they were already .tif, this will convert them to ImageJ .tif).

# Using Domino Denoise on your colour images, stacks and hyperstacks

To run on anything other than 2D grayscale images, use DD_4D.py. This supports an arbitrary number of dimensions, as long as the last two dimensions are x and y. For example, here we use it on a 16x6x2x250x250 (tzcxy) image:
  
```python
cd <masterdirectoryname>
conda activate DD
python DD_4D.py livecells
```  

Output is in ImageJ format.

# Using Domino Denoise on provided datasets

To run DD on the noisy microscope images, open a terminal in the master directory and run:

```python
cd <masterdirectoryname>
python DD.py Microscope_gaussianpoisson
```
The denoised results will be in the directory 'Microscope_gaussianpoisson_N2F'.

To run DD on our other datasets we first need to add synthetic gasussian noise. For example to test DD on Set12 with sigma=25 gaussian noise, we would first: 
```python
cd <masterdirectoryname>
python add_gaussian_noise.py Set12 25
```
This will create the folder 'Set12_gaussian25' which we can now denoise:

```python
python DD.py Set12_gaussian25
```
Which returns the denoised results in a folder named 'Set12_gaussian25_DD'.
  


# Calculate accuracy of Domino Denoise

To find the PSNR and SSIM between a folder containing denoised results and the corresponding folder containing known ground truths (e.g. Set12_gaussian25_DD and Set12 if you followed above), we need to install one more conda package:

```python
conda activate DD
conda install -c anaconda scikit-image=0.19.2
```

Now we measure accuracy with the code:
```terminal
cd <masterdirectoryname>
python compute_psnr_ssim.py Set12_gaussian25_DD Set12 255
```

You can replace 'Set12' and 'Set12_gaussian25' with any pair of denoised/ground truth folders (order doesn't matter). Average PSNR and SSIM will be returned for the entire set.

The '255' at the end denotes the dynamic range of the image, in the case of the 8-bit images from Set12, '255' is a sensible value. For the Microscope data, '700' is a more sensible value and will replicate the results from our paper.
  

  
# Running compared methods

We can run DIP, Noise2Self, P2S and N2F+DOM in the DD environment:

```python
conda activate DD
python DIP.py Microscope_gaussianpoisson
python N2S.py Microscope_gaussianpoisson
python P2S.py Microscope_gaussianpoisson
python N2FDOM.py Microscope_gaussianpoisson
```

