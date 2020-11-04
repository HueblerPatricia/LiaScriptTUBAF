<!--

author:   Patricia Huebler

email:    Patricia.Huebler@student.tu-freiberg.de

language: en

narrator: UK English Female

import:  https://github.com/LiaTemplates/Pyodide
-->

# Digital image processing in Python 3 - part II: pseudocolors and histogram modification

See on LiaScript:

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/PseudocolorsHistograms.md


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
For a colorful image we need, of course, some colors. Let's define some more today to get a bit more interesting histograms later.

```python
red           = [255,0  ,0  ]
red_violet    = [255,0  ,64 ]
red_wine      = [191,0  ,26 ]
dark_red      = [64 ,0  ,0  ]
green         = [0  ,255,0  ]
blue          = [0  ,0  ,255]
dark_blue     = [0  ,0  ,124]
bright_blue   = [26 ,26 ,255]
brown         = [145,111,124]
purple        = [255,0  ,255]
violet        = [126,0  ,255]
yellow        = [255,255,0  ]
deep_yellow   = [255,166,0  ]
orange        = [255,212,45 ]
cyan          = [0  ,255,255]
blue_green    = [0  ,255,128]
blueish_green = [0  ,255,64 ]
white         = [255,255,255]
black         = [0  ,0  ,0  ]
gray1         = [100,100,100]
gray2         = [99 ,99 ,99 ]
gray3         = [101,101,101]

```
@Pyodide.eval

  {{1}}
**************************************************************
--{{1}}--
Now we create an array with 3 color layers to be our test image.

```python
def rotateList(lst, rot):

    '''Rotates a given list

    Parameters
    ----------
    lst : list
        list with entries of any data type
    rot : integer
        number of entries for the list to be rotated

    Returns
    -------
    list
        the rotated list
    '''
    l2 = lst[rot:] + lst[:rot]
    return l2

# some lists with colors
colorList = [yellow, white, red, orange, purple]
colorList_spread = [gray1, gray2, gray3]
colorList_red = [red, red_wine, dark_red, yellow, deep_yellow, orange, brown]
colorList_cyan = [green, blue, cyan, blue_green, blueish_green, dark_blue, bright_blue]

```
@Pyodide.eval

**************************************************************

  {{2}}
**************************************************************
--{{2}}--
To look at some pseudo colors and single color layers let's take an example with some more colors like this list named "color list cyan".

```python
colorList = colorList_cyan
```
@Pyodide.eval
**************************************************************

#### Draw our picture

--{{0}}--
Today we define a function for that drawing because we want to use it several times in this script.

```python
#define the picture's side length
pictSize = 120

def drawSquares(colorList, pictSize):

    '''creates a square shaped picture from colored squares

    Parameters
    ----------
    colorList : list
        list of rgb colors to use
    pictSize : integer
        side length of the future picture

    Returns
    -------
    2D array
        the ready and normed kernel
    '''

    pictSize = pictSize//len(colorList)
    squareLen = pictSize
    pictSize *= len(colorList)
    pictarray = np.zeros([pictSize, pictSize, 3], dtype=np.uint8) #3 layers for r,g,b


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
    return pictarray

```
@Pyodide.eval

  {{1}}
***************************************************
--{{1}}--
And now the function call and a plot of the picture.

```python
pictarray = drawSquares(colorList, pictSize)

fig, ax = plt.subplots()
plt.imshow(pictarray)
plt.show()

plot(fig)

```
@Pyodide.eval
***************************************************

### Shades of gray

--{{0}}--
Again we have to define the function ourselves because LiaScript doesn't know the necessary Python modules for importing the function.

See "Digital Image Filters" for explanation:

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/DigitalImageFilters.md

```python
def rgb2gray(rgb):

    '''Converts an rgb pixel into grayscale

    Parameters
    ----------
    rgb : array
        a pixel with 3 color layers

    Returns
    -------
    float
        the grayscale value for the given rgb
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
```
@Pyodide.eval

> **Remark:** You may also change the weights for the 3 colors. Those above were taken from Wikipedia (https://de.wikipedia.org/wiki/Grauwert).

  {{1}}
*********************************************************
--{{1}}--
Here some different settings were used for plotting to make sure that the plotting function doesn't perform a histogram normalization automatically.

Now the computation:

```python
gray = rgb2gray(pictarray)

fig,ax = plt.subplots()
cmap = plt.get_cmap('gray')
plt.imshow(gray, norm=None, vmin=0, vmax=255, cmap = cmap)
plt.show()

plot(fig)

```
@Pyodide.eval

*********************************************************

### Compare the histograms of the color and the grayscale image

```python
flatpict = pictarray.flatten() #flatten() transforms an array of multi dimensions into 1D
flatgray = gray.flatten()

fig = plt.figure(figsize = (8,6))
sub1 = fig.add_subplot(2,2,1)
sub1.hist(flatpict)
sub1.set_title('Histogram of the color image')
sub1.set_xlabel('colors')
sub1.set_ylabel('frequency')
sub2 = fig.add_subplot(2,2,2)
sub2.hist(flatgray)
sub2.set_title('Histogram of the grayscale image')
sub2.set_xlabel('colors')
plt.tight_layout()

plot(fig)

```
@Pyodide.eval

--{{0}}--
You see that there are more bins in the color image's histogram. By converting into grayscale some of the colors become very similar grayscale values and end up in the same bin.

## Pseudo colors

--{{0}}--
Pseudo colors are used to show special aspects of a picture. The easiest way to create some pseudo colors is to plot every color layer separately. But how do we get single color layers? See below.

```python
layer0_img = pictarray[:,:,0]
layer1_img = pictarray[:,:,1]
layer2_img = pictarray[:,:,2]

```
@Pyodide.eval

  {{1}}
******************************************************
--{{1}}--
You may plot all color layers with the same colormap settings if you want. In this example they were dyed in the color they really show in rgb, that means red, green and blue. See Python documentation for other colormaps.

```python
fig = plt.figure(figsize=(8,8))
sub1 = plt.subplot(3, 3, 1)
sub1.imshow(layer0_img, cmap = 'Reds')
sub2 = plt.subplot(3, 3, 2)
sub2.imshow(layer1_img, cmap = 'Greens')
sub3 = plt.subplot(3, 3, 3)
sub3.imshow(layer2_img, cmap = 'Blues')

plot(fig)

```
@Pyodide.eval
*****************************************************

## Histogram modification

--{{0}}--
What is a histogram? That seems to be a very elementary question, but you have to be sure about the answer if you want to do histogram modification. Let's have a look at a random example.

literature for this chapter:  Gonzalez, Woods: Digital Image Processing. Third Edition. PHI Learning Private Limited. New Delhi, 2008

### What is a histogram?

--{{0}}--
A histogram is a plot of a frequency distribution. When we talk about images that means a histogram shows how many pixel are of a special quality, in our case of a special color or amount of colors, because our bins normally contain more than one color.

```python
mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000)

# the histogram of the data
fig, ax = plt.subplots()
plt.hist(x, 50, density=1, facecolor='blue', alpha=0.75)

plot(fig)

```
@Pyodide.eval

### Histogram equalization

--{{0}}--
Histogram equalization is a method of contrast adjustment. We take all colors that are already contained in the picture and distribute them equally. We don't loose colors, we just change them. It is a common method for increasing contrast in a whole image.

In histogram equalization we compute the following:

$s_{k} = (L - 1)\sum_{j=0}^{k} p_{r}r_{j} = \dfrac{L - 1}{MN}\sum_{j=0}^{k}n_{j}$

What do the variables mean?
| variable | meaning                                |
|----------|----------------------------------------|
| $s_{k}$  | the new color value                    |
| $L$      | amount of occurring colors             |
| $p_{r}$  | their probability of occurrence        |
| $r_{j}$  | a single color                         |
| $M$      | picture's row dimension                |
| $N$      | picture's column dimension             |
| $n_{i}$  | total count of pixels of one color     |

source for formula: Gonzalez, Woods: Digital Image Processing. Third Edition. PHI Learning Private Limited. New Delhi 2008, page 126

#### Function for histogram equalization

... just an implementation of the formula.

```python

def equalhist(pict, equalpict):

    '''does a histogram equalization

    Parameters
    ----------
    pict : 2D array of numbers
       an image of grayscale values

    equalpict : 2D array of numbers
       the (empty) picture to write in

    Returns
    -------
    2D array
       the modified grayscale image
    '''

    flat = pict.flatten()
    hist = np.histogram(flat, bins=256, density=False)[0]
    x,y = np.shape(pict)
    probs = hist / (x*y)
    factors = probs
    s = 0
    i = 0
    for elem in probs:
        s += 255*elem
        factors[i] = int(s+0.5) # round
        i += 1

    for i in range(0,x):
            for j in range(0,y):
                value = int(pict[i][j]+0.5)
                new_value = factors[value]
                equalpict[i][j] = new_value
    return equalpict

```
@Pyodide.eval

  {{1}}
******************************************************
Let's try out!

```python
gray_equal = gray.copy()
gray_equal = equalhist(gray, gray_equal)

fig,ax = plt.subplots()
cmap = plt.get_cmap('gray')
plt.imshow(gray_equal, norm=None, vmin=0, vmax=255, cmap = cmap)

plot(fig)

```
@Pyodide.eval
******************************************************

#### Histogram comparison
--{{0}}--
Now we can take the histograms and see what our function has done.

```python
flatgray = gray.flatten()
flatequal = gray_equal.flatten()

fig = plt.figure(figsize = (8,6))
sub1 = fig.add_subplot(2,2,1)
sub1.hist(flatgray)
sub1.set_title('Before')
sub1.set_xlabel('colors')
sub1.set_ylabel('frequency')
sub2 = fig.add_subplot(2,2,2)
sub2.hist(flatequal)
sub2.set_title('After')
sub2.set_xlabel('colors')

plot(fig)

```
@Pyodide.eval

--{{0}}--
Now we see that the colors are not equally distributed. That is not a problem of the formula, it is a problem of the image consisting of so few colors. In fact that works better with a picture that has at least an amount of 30 colors which occur in a kind of Gaussian distribution.

### Histogram spreading

--{{0}}--
Histogram spreading is another method for increasing contrast. Here we define an interval of colors and spread them up into the whole grayscale values from zero to two hundred fifty five. Within this method we don't change any possibility for a color occurring, we just shift the color values.

#### Another test image

--{{0}}--
Are the squares visible for you?

```python
colorList = colorList_spread
pictarray2 = drawSquares(colorList, pictSize)

fig, ax = plt.subplots()
plt.imshow(pictarray2)
plt.show()

plot(fig)

```
@Pyodide.eval

  {{1}}
*******************************************************
--{{1}}--
Again we convert it into grayscale to work with only one color layer. But that should look the same, because in rgb we used defined variants of gray. This time it is important to use the plot settings below, because otherwise the plot command would do the histogram spreading for us, but in a way we could not go on working with them.

```python
gray2 = rgb2gray(pictarray2)

fig,ax = plt.subplots()
cmap = plt.get_cmap('gray')
plt.imshow(gray2, norm=None, vmin=0, vmax=255, cmap = cmap)
plt.show()

plot(fig)

```
@Pyodide.eval

*******************************************************

#### Histogram spreading explained

--{{0}}--
This is the histogram of our grayscale image. You see that there are only three different colors and they are very close to each other.

```python
flatgray2 = gray2.flatten()

fig,ax = plt.subplots()
plt.hist(flatgray2)
plt.show()

plot(fig)

```
@Pyodide.eval

  {{1}}
********************************************************
--{{1}}--
 So what do we want to do? We want to take the two bins at the minimum and maximum color value and pull them away from each other to the ends of the whole color interval. When this is done we have to define all color values between them new by there former distance to the minimum value. We could also do the computation with the maximum value, that does not matter.

```python

def spreadhist(pict, flat, vmax = 255, vmin = 0):

    '''does a histogram spreading

    Parameters
    ----------   
    pict : 2D array of numbers
        an image of grayscale values

    flat : 1D array of numbers
        the one dimensional variant of the picture

    vmax : number
        the future maximum color value

    vmin : number
        the future minimum color value

    Returns
    -------
    2D array
        the modified grayscale image
    '''

    min_color = min(flat)
    max_color = max(flat)
    diff_old = max_color - min_color
    diff_new = vmax - vmin
    factor = diff_new / diff_old
    x,y = np.shape(pict)
    for i in range(0,x):
        for j in range(0,y):
            diff = pict[i][j] - min_color
            pict[i][j] = diff * factor
    return pict

```
@Pyodide.eval

```python
gray_spread2 = spreadhist(gray2, flatgray2)

fig,ax = plt.subplots()
cmap = plt.get_cmap('gray')
plt.imshow(gray_spread2, norm=None, vmin=0, vmax=255, cmap = cmap)
plt.show()

plot(fig)

```
@Pyodide.eval

********************************************************

### Comparison of all histograms

--{{0}}--
To have a direct comparison we need to do the histogram spreading also with the picture that was created first.

```python
gray_spread = spreadhist(gray, flatgray)
flatspread = gray_spread.flatten()
```
@Pyodide.eval

  {{1}}
**************************************************
--{{1}}--
You see that histogram spreading works fine while equalization does not really do what we imagined. That is no problem of the function but of our picture only consisting of few colors.

```python

fig = plt.figure(figsize=(12,12))
sub1 = fig.add_subplot(3,3,1)
sub1.hist(flatgray)
sub1.set_title('The original')
sub1.set_xlabel('colors')
sub1.set_ylabel('frequency')
sub2 = fig.add_subplot(3,3,2)
sub2.hist(flatequal)
sub2.set_title('After histogram equalization')
sub2.set_xlabel('colors')
sub2 = fig.add_subplot(3,3,3)
sub2.hist(flatequal)
sub2.set_title('After histogram spreading')
sub2.set_xlabel('colors')

plot(fig)

```
@Pyodide.eval

**************************************************

### Better example for histogram equalization

--{{0}}--
On the following slides we will see the same process of histogram equalization but with a different picture. So let's do the familiar processes quickly.

```python
gray0   = [0  ,0  ,0  ]
gray1   = [10 ,10 ,10 ]
gray2   = [20 ,20 ,20 ]
gray3   = [30 ,30 ,30 ]
gray4   = [40 ,40 ,40 ]
gray5   = [50 ,50 ,50 ]
gray6   = [60 ,60 ,60 ]
gray7   = [70 ,70 ,70 ]
gray8   = [80 ,80 ,80 ]
gray85  = [85 ,85 ,85 ]
gray9   = [90 ,90 ,90 ]
gray95  = [95 ,95 ,95 ]
gray10  = [100,100,100]
gray102 = [102,102,102]
gray105 = [105,105,105]
gray107 = [107,107,107]
gray11  = [110,110,110]
gray112 = [112,112,112]
gray114 = [114,114,114]
gray115 = [115,115,115]
gray116 = [116,116,116]
gray117 = [117,117,117]
gray118 = [118,118,118]
gray119 = [119,119,119]
gray12  = [120,120,120]
gray121 = [121,121,121]
gray122 = [122,122,122]
gray123 = [123,123,123]
gray124 = [124,124,124]
gray125 = [125,125,125]
gray126 = [126,126,126]
gray127 = [127,127,127]
gray128 = [128,128,128]
gray129 = [129,129,129]
gray13  = [130,130,130]
gray131 = [131,131,131]
gray132 = [132,132,132]
gray135 = [135,135,135]
gray137 = [137,137,137]
gray14  = [140,140,140]
gray142 = [142,142,142]
gray145 = [145,145,145]
gray15  = [150,150,150]
gray155 = [155,155,155]
gray16  = [160,160,160]
gray165 = [165,165,165]
gray17  = [170,170,170]
gray18  = [180,180,180]
gray19  = [190,190,190]
gray20  = [200,200,200]
gray21  = [210,210,210]
gray22  = [220,220,220]
gray23  = [230,230,230]
gray24  = [240,240,240]
gray25  = [250,250,250]

colorList_gray = [gray0,
                  gray1,
                  gray2,
                  gray3,
                  gray4,
                  gray5,
                  gray6,
                  gray7,
                  gray8,
                  gray85,
                  gray9,
                  gray95,
                  gray10,
                  gray102,
                  gray105,
                  gray107,
                  gray11,
                  gray112,
                  gray114,
                  gray115,
                  gray116,
                  gray117,
                  gray118,
                  gray119,
                  gray12,
                  gray121,
                  gray122,
                  gray123,
                  gray124,
                  gray125,
                  gray126,
                  gray127,
                  gray128,
                  gray129,
                  gray13,
                  gray131,
                  gray132,
                  gray135,
                  gray137,
                  gray14,
                  gray142,
                  gray145,
                  gray15,
                  gray155,
                  gray16,
                  gray165,
                  gray17,
                  gray18,
                  gray19,
                  gray20,
                  gray21,
                  gray22,
                  gray23,
                  gray24,
                  gray25]

```
@Pyodide.eval

* new color list -> done
+ creating the image -> left to do
+ converting to grayscale -> left to do
+ histogram equalization -> left to do
- compare the histograms -> left to do


  {{1}}
**************************************************

```python
colorList = colorList_gray
pictarray = drawSquares(colorList, pictSize)

fig,ax = plt.subplots()
cmap = plt.get_cmap('gray')
plt.imshow(pictarray, norm=None, vmin=0, vmax=255, cmap = cmap)
plt.show()

plot(fig)

```
@Pyodide.eval

* new color list -> done
+ creating the image -> done
+ converting to grayscale -> left to do
+ histogram equalization -> left to do
- compare the histograms -> left to do

**************************************************

  {{2}}
**************************************************

```python
gray = rgb2gray(pictarray)

```
@Pyodide.eval

* new color list -> done
+ creating the image -> done
+ converting to grayscale -> done
+ histogram equalization -> left to do
- compare the histograms -> left to do

**************************************************

  {{3}}
**************************************************

```python
gray_equal = gray.copy()
gray_equal.fill(255)
gray_equal = equalhist(gray, gray_equal)

```
@Pyodide.eval

* new color list -> done
+ creating the image -> done
+ converting to grayscale -> done
+ histogram equalization -> done
- compare the histograms -> left to do

**************************************************

  {{4}}
**************************************************

```python
flatgray = gray.flatten()
flatequal = gray_equal.flatten()

fig = plt.figure(figsize = (8,6))
sub1 = fig.add_subplot(2,2,1)
sub1.hist(flatgray)
sub1.set_title('Before')
sub1.set_xlabel('colors')
sub1.set_ylabel('frequency')
sub2 = fig.add_subplot(2,2,2)
sub2.hist(flatequal)
sub2.set_title('After')
sub2.set_xlabel('colors')

plot(fig)

```
@Pyodide.eval

**************************************************

## More opportunities

Of course, that was a very short introduction to the topic. If you are interested in really dealing with histogram modification I recommend using the Python module "skimage". There you have a lot of filters, histogram modifications and much more, all ready implemented.
Right now LiaScript is not yet able to include "skimage" or other modules suitable for the topic, like for examle "PIL". If you are interested in the topic, just take a Python editor and try out what the language offers you!  

