<!--

author:   Patricia Huebler

email:    Patricia.Huebler@student.tu-freiberg.de

language: en

import:  https://github.com/LiaTemplates/Pyodide
-->

# The basics of digital image filters in Python 3

See on LiaScript:\newline
 https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/DigitalImageFilters.md

 
## Goals and fields of application
| goals                  | fields               |
|------------------------|----------------------|
| reduce noise           | satellite pictures   |
| make details stand out | aerial photos        |
|                        | CT / X-ray imaging   |

What do we want to do?
* prepare image for further processing
- improve image for human user
--{{0}}--
First of all let's make clear, what we use filter methods for.
Basically there are two possibilities for that: In the first one, we want to change a picture in that way, that a computer is able to do further work on it.
In the second one, which we want to deal with, our goal is to improve the image for a human user. That means, we want to make special parts of the picture more or better visible.

## Preparations
Import necessary modules

```python
import numpy as np                #working with arrays
import matplotlib.pyplot as plt   #plot an image

```
@Pyodide.eval

--{{0}}--
First we need to bind in some Python modules.

What are the modules for?
| module            | content                                    |
|-------------------|--------------------------------------------|
| NumPy             | work with arrays, matrices, vectors etc.   |
| Matplotlib.Pyplot | plotting images and referred settings      |

### Create a simple picture
#### Colorful - rgb
--{{0}}--
For a colorful image we need, of course, some colors. Let's define them.

```python
red       = [255,0  ,0  ]
green     = [0  ,255,0  ]
blue      = [0  ,0  ,255]
brown     = [145,111,124]
purple    = [255,0  ,255]
yellow    = [255,255,0  ]
orange    = [255,212,45 ]
white     = [255,255,255]
turquoise = [0  ,255,255]
black     = [0  ,0  ,0  ]
```
@Pyodide.eval

#### Creating our picture
--{{0}}--
Now we create an array with 3 color layers. You can take this example, play with the colors or be more creative.

```python
def rotateList(lst, rot):
    l2 = lst[rot:] + lst[:rot]
    return l2

#the picture's side length
pictSize = 120

# a list with predefined colors from above
# create list because we want to "play" with the colors
colorList = [yellow, white, red, orange, purple]

pictSize = pictSize//len(colorList)
squareLen = pictSize
pictSize *= len(colorList)
#create array with 3 color layers for r, g and b
pictarray = np.zeros([pictSize, pictSize, 3], dtype=np.uint8)

lsta = 0
lstp = squareLen
for lines in range(0,len(colorList)):
    csta = 0
    cstp = squareLen
    for col in range(0,len(colorList)):
        pictarray[lsta:lstp,csta:cstp] = colorList[col]
        csta += squareLen
        cstp += squareLen
    lsta += squareLen
    lstp += squareLen
    colorList = rotateList(colorList,-2)


```
@Pyodide.eval

#### Colorful squares

```python
plt.figure(dpi=180)
plt.imshow(pict)
plt.show()
```

Normally you could use the commands above, but again we are in LiaScript, so use the lines beyond!

```python
fig, ax = plt.subplots()
plt.imshow(pictarray)
plt.show()

plot(fig)
```
@Pyodide.eval

### Shades of gray
--{{0}}--
You may do all following computations with all three color layers or you may convert your picture to grayscale.

Normally you would have another ready implemented function for this and could use the following code:

```python
from skimage.color import rgb2gray

gray = rgb2gray(pictarray)

#plt.imshow(gray, cmap = plt.get_cmap('gray'))
#plt.show()
```

But LiaScript doesn't know the module "skimage", so let's write our own function "rgb2gray". Anyway, for understanding that is the better way.

### Change to grayscale yourself

```python
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
```
@Pyodide.eval

Remark: You may also change the weights for the 3 colors. Those above were taken from Wikipedia (https://de.wikipedia.org/wiki/Grauwert).

Now the computation:

```python
gray = rgb2gray(pictarray)
```
@Pyodide.eval

### Look at "gray"

```python
fig, ax = plt.subplots()
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()
plot(fig)
```
@Pyodide.eval

## Theory of linear filters
Literature for this chapter: Gonzalez, Woods: Digital Image Processing. Third Edition. PHI Learning Private Limited. New Delhi, 2008. Chapter 3.4

## Image as array
--{{0}}--
The table represents our grayscale image. Each field is one pixel marked with "p".

| p | p | p | p | p | p |
| p | p | p | p | p | p |
| p | p | p | p | p | p |
| p | p | p | p | p | p |
| p | p | p | p | p | p |
| p | p | p | p | p | p |
| p | p | p | p | p | p |

### Filter mask
--{{0}}--
You see the rectangle marked by an "k" in every field. That is our filter mask or kernel. That is normally a square with an odd side length. Why odd? Because we want to work easily with integers.

| p | p | p | p | p | p |
| k | k | k | p | p | p |
| k | k | k | p | p | p |
| k | k | k | p | p | p |
| p | p | p | p | p | p |
| p | p | p | p | p | p |
| p | p | p | p | p | p |

### Current pixel
--{{0}}--
With the picture and the kernel we want to calculate the new color value for the pixel in the middle of the kernel marked with an "x". If this operation is linear, we call the whole thing "linear filter".

| p | p | p | p | p | p |
| k | k | k | p | p | p |
| k | x | k | p | p | p |
| k | k | k | p | p | p |
| p | p | p | p | p | p |
| p | p | p | p | p | p |
| p | p | p | p | p | p |

New value of the current pixel "x" computed by using the following formula:

$new value = \dfrac{1}{m\cdot n}\cdot\sum_{i=1,j=1}^{m,n} value picture[i][j]\cdot value kernel[i][j] $

Translated into Python:

```python
def convolve(kernel,pictpart):
    s = 0.0
    temporal_array = np.multiply(kernel,pictpart)
    s = sum(temporal_array)
    s = sum(s)
    return s
```
@Pyodide.eval

## One example of linear image filters
Remark: All following examples are ready implemented in "OpenCV". (see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html). Here working on rgb pictures is as easy as working on grayscale images.

### Average filter
The kernel is a square nxn and all entries are of the same weight.

```python
n = 5
kernel=np.ones((n,n), dtype=np.float32)
kernel.fill(1/(n*n))
print(kernel)
```
@Pyodide.eval

#### New picture
Create empty picture to write our blurred picture values. It must have the same size as the original (grayscale) one, so you may copy it and erase it's values.

```python
newpict = gray[0:][0:].copy() #copy rows and columns of "gray"
newpict.fill(0)

```
@Pyodide.eval

Have a look if you want.

```python
fig, ax = plt.subplots()
plt.imshow(newpict, cmap = plt.get_cmap('gray'))
plt.show()
plot(fig)
```
@Pyodide.eval

#### "Pictpart"
Look into the formulas above. We need
* an original (grayscale) image        -> done
+ a kernel                             -> done
+ a new, empty picture                 -> done
+ a picture part (same size as kernel) -> left to do!

```python
pictpart = kernel.copy()
pictpart.fill(0)
```
@Pyodide.eval

### The "real" filter process
--{{0}}--
Imagine you have your picture and the filter mask. Now you move the filter mask step by step over the original picture, calculate the new color values for every pixel that is located in the middle of the filter mask. And you write those new color values into the new, empty image.

--{{0}}--
This process is the same for all linear filters and in fact for most filters at all.

```python
for i in range(0+(kernel.shape[0]//2),  len(gray)-(kernel.shape[0])):
    for j in range(0+(kernel.shape[1]//2),  len(gray[1])-(kernel.shape[1])):

        for k in range(0,  (kernel.shape[0])):
            for l in range(0,  (kernel.shape[1])):
                pictpart[k][l]=gray[i+k][j+l]
        newpict[i+(kernel.shape[0]//2)][j+(kernel.shape[1]//2)]=convolve(kernel, pictpart)

```
@Pyodide.eval

### See the blurred image

```python
fig, ax = plt.subplots()
plt.imshow(newpict, cmap = plt.get_cmap('gray'))
plt.show()
plot(fig)
```
@Pyodide.eval

## Further computations
Some ideas about what to do with blurred images.

### Differrence picture
--{{0}}--
Now we have the original grayscale image and the blurred one. What can we do with them? Subtract one from the other!

```python
diffPict = gray - newpict
fig, ax = plt.subplots()
plt.imshow(diffPict, cmap = plt.get_cmap('gray'))
plt.show()
plot(fig)
```
@Pyodide.eval

Now we see where was blurred most because that is visible best now. Especially we see edges.

### Unsharp masking
Sometimes we want to sharpen edges. So let's add the "diffPict" to the grayscale one. This process is called "unsharp masking".

```python
masked = gray + diffPict
fig, ax = plt.subplots()
plt.imshow(masked, cmap = plt.get_cmap('gray'))
plt.show()
plot(fig)
```
@Pyodide.eval

### Other filters
Just a short overview. The mentioned book "Digital image processing" is a good resource to go on reading.

#### Linear filter: Gaussian filter
This one works like the average filter. It's only difference is the kernel. Here you don't have always the same value, but something like a Gaussian bell curve in 2D. You may use any ready implemented function or you take the computation steps from above after creating your kernel by the following lines:

```python
def create_gaussiankernel(site=5, sigma=1):
    ax = np.arange(-site // 2 + 1., site // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

kernel = create_gaussiankernel(5,5)
```

#### Nonlinear filters
**Median filter:** You do also have a kind of kernel, but it is not meant to calculate values with linear algebra. The process works the following: You collect all color values in the kernel's range of the original image and sort them by size. Then you choose the one right in the middle of your list as the new color value of the current pixel.

**Minimum and maximum filter:** Minimum and maximum filter work in a very similar way: You do also sort your values, but choose the lowest or highest number as new color value.

## Further ideas
By unsharp masking we have sharpened the edges but lost contrast. We could go on with some histogram modification to improve our "masked" image.
Remark: Als implemented in OpenCV. See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
