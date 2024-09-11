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
```

Now follow the instructions here to get a command you can enter to install pytorch: https://pytorch.org/get-started/locally/. If you have a GPU, select one of the compute platforms that starts with 'CUDA'. The command the website spits out should start with 'pip3'. Enter that command into the terminal and press enter, then once it's installed proceed as follows to install some additonal needed libraries:

```python
conda install conda-forge::tifffile
conda install anaconda::pandas
conda install anaconda::scipy
```

If the installs don't work, removing the specific version may fix this. To do this, remove everything after the equals sign, including the equals sign (e.g. conda install pytorch::pytorch).

# Using debleeder on IMC and other highly multiplexed data
(Note: The program expects tiff stacks for input IMC data. If your data is saved as a sequence of individual channel files, open ImageJ and go File->Import->Image Sequence, select the folder containing the individual channels and click Open. Once open go Image->Stacks->Images to Stack and then save the resulting image stack, this file should work with RefineOT.)

To run the debleeder create a folder in the master directory (the directory that contains debleed.py) and put your raw IMC images into it. Then open anaconda prompt/terminal and run the following:

```python
cd <masterdirectoryname>
conda activate DD
python debleed.py <imcfolder>/<imcfilename> <channel_to_debleed>
```

Replacing "masterdirectoryname" with the full path to the directory that contains debleed.py. For example, to apply this to the 21st channel of the IMC_smallcrop data (using 1-indexing) included in this repository we would run:

```python
cd <masterdirectoryname>
conda activate DD
python debleed.py IMC_smallcrop/IMC_smallcrop.tif 21
```

For best results on IMC, you should supply a veto matrix of channels you do not want to be considered when debleeding the target channel. For format, see IMC_smallcrop_withcsv/IMC_smallcrop.csv. Essentially the columns and rows list each channel, and a 0 in (x,y) indicates that column x's channel will NOT be considered when debleeding the row y's channel. 

This might be done, for example if it is a prioi known which channels are suceptible to bleed through into other channels, or if it is known certain channels contain legitimately similar signal that is not the result of bleed through. Ultimately you should put as much information as is known into this matrix to achieve optimal image restoration. The names of columns and rows in the .csv file is irrelevant and not read by the program, it will assume the fouth row corresponds to the fourth channel etc., so if you do name the columns and rows ensure they correspond to the order in which they appear in the tiff stack. The program automatically detects the presence of a veto matrix (just give it the same name as the target tiff file, but ending in .csv), so you can simply run:

```python
cd <masterdirectoryname>
conda activate DD
python debleed.py IMC_smallcrop_withcsv/IMC_smallcrop.tif 21
```

The denoiser and debleeder/denoiser combo can be run in the exact same way:

```python
cd <masterdirectoryname>
conda activate DD
python debleed_and_denoise.py IMC_smallcrop_withcsv/IMC_smallcrop.tif 21
python denoise.py IMC_smallcrop/IMC_smallcrop.tif 21
```


# Using DD on your 2D grayscale data

Create a folder in the master directory (the directory that contains debleed.py) and put your noisy images into it. Then open anaconda prompt/terminal and run the following:

```python
cd <masterdirectoryname>
conda activate DD
python DD.py <noisyfolder>/<noisyimagename>
```
Replacing "masterdirectoryname" with the full path to the directory that contains DD.py, replacing "noisyfolder" with the name of the folder containing images you want denoised and replacing "noisyimagename" with the name of the image file you want denoised. Results will be saved to the directory '<noisyolder>_denoised'. Issues may arise if using an image format that is not supported by the tifffile python package, to fix these issues you can open your images in ImageJ and re-save them as .tif (even if they were already .tif, this will convert them to ImageJ .tif).

# Reproducibility

To run anything beyond this point in the readme, we need to install another conda library:

```python
conda install anaconda::scikit-image=0.23.2
```

# Using DD on provided datasets

To run DD on one of the noisy microscope images, open a terminal in the master directory and run:

```python
cd <masterdirectoryname>
python denoise2D.py Microscope_gaussianpoisson/1.tif
```
The denoised results will be in the directory 'Microscope_gaussianpoisson_denoised'.

To run DD on our other datasets we first need to add synthetic gasussian noise. For example to test DD on Set12 with sigma=25 gaussian noise, we would first: 
```python
cd <masterdirectoryname>
python add_gaussian_noise.py Set12 25
```
This will create the folder 'Set12_gaussian25' which we can now denoise:

```python
python denoise2D.py Set12_gaussian25/01.tif
```
Which returns the denoised results in a folder named 'Set12_gaussian25_denoised'.
  


# Calculate accuracy of DD

To find the PSNR and SSIM between a folder containing denoised results and the corresponding folder containing known ground truths (e.g. Set12_gaussian25_denoised and Set12 if you followed above), we need to install one more conda package:

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

# Reproducing Figure 1 results

To track the PSNR over time of RefineOT denoise, do the following:

```python
cd <masterdirectoryname>
cd Fig1
python add_gaussian_noise.py 345 25
python TrackPSNR.py
```

