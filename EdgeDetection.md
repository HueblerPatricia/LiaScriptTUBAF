<!--

author:   Patricia Huebler

email:    Patricia.Huebler@student.tu-freiberg.de

language: en

narrator: UK English Female

import:  https://github.com/LiaTemplates/Pyodide
-->

# Digital image processing in Python 3 - part III: edge and point detection

In this chapter we want to make color gradients visible. We are looking for edges and points in a picture.


See on LiaScript:

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/EdgeDetection.md

## Preparations - import Python modules

```python
import numpy as np                #working with arrays
import matplotlib.pyplot as plt   #plot an image

```
@Pyodide.eval


What are the modules for?
| module            | content                                    |
|-------------------|--------------------------------------------|
| NumPy             | work with arrays, matrices, vectors etc.   |
| Matplotlib.Pyplot | plotting images and referred settings      |

### Create a simple picture

See

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/DigitalImageFilters.md

and

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/PseudocolorsHistograms.md

for explanation in this chapter.

#### Colorful - rgb
Define some colors for the image.

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

```
@Pyodide.eval

  {{1}}
**************************************************************

Create an array with 3 color layers to be our test image.

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
colorList_red = [red, red_wine, dark_red, yellow, deep_yellow, orange, brown]
colorList_cyan = [green, blue, cyan, blue_green, blueish_green, dark_blue, bright_blue]

```
@Pyodide.eval

**************************************************************

  {{2}}
**************************************************************

Choose a color list or define one yourself.

```python
colorList = colorList_cyan
```
@Pyodide.eval

**************************************************************

#### Draw our picture

A function for drawing the image.

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

Function call and a plot of the picture.

```python
pictarray = drawSquares(colorList, pictSize)

fig, ax = plt.subplots()
plt.imshow(pictarray)
plt.show()

plot(fig)

```
@Pyodide.eval

***************************************************

### Grayscale

Just work with one color layer to make things easier.

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

> **Remark:** You may change the weights for the 3 colors. Those above were taken from Wikipedia (https://de.wikipedia.org/wiki/Grauwert).

  {{1}}
***************************************************

Now the computation and a plot:

```python
gray = rgb2gray(pictarray)

fig, ax = plt.subplots()
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()
plot(fig)
```
@Pyodide.eval

***************************************************

## Edge detection
Literature for this chapter: Gonzalez, Woods: Digital Image Processing. Third Edition. PHI Learning Private Limited. New Delhi, 2008. Chapter 10

> **Remark:** Edge detection is nothing but linear image filtering.

For theory of linear filters see

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/DigitalImageFilters.md

Here just in short.

For more explanation, see:

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/DigitalImageFilters.md

  {{1}}
***************************************************

Convolution:

$new value = \dfrac{1}{m\cdot n}\cdot\sum_{i=1,j=1}^{m,n} value picture[i][j]\cdot value kernel[i][j] $


```python
def convolve(kernel,pictpart):

    '''convolution in 2D

    Parameters
    ----------
    kernel : 2D array of numbers
       the filter mask to use

    pictpart : 2D array of numbers
       currently active part of the picture


    Returns
    -------
    float
       the new color value
    '''

    s = 0.0
    temparr=np.multiply(kernel,pictpart)
    s = sum(temparr)
    s = sum(s)
    return s

```
@Pyodide.eval

***************************************************

  {{2}}
***************************************************

Filtering:

```python
def filter_image(pict, kernel):

    '''image filtering

    Parameters
    ----------
    pict : 2D array of numbers
       an image of grayscale values

    kernel : 2D array of numbers
       the filter mask to use

    Returns
    -------
    2D array
       the modified grayscale image
    '''

    newpict = pict[0:][0:].copy()
    newpict.fill(0)

    pictpart = kernel.copy()
    pictpart.fill(0)

    for i in range(0+(kernel.shape[0]//2),  len(pict)-(kernel.shape[0])):
        for j in range(0+(kernel.shape[1]//2),  len(pict[1])-(kernel.shape[1])):

            for k in range(0,  (kernel.shape[0])):
                for l in range(0,  (kernel.shape[1])):
                    pictpart[k][l]=pict[i+k][j+l]
            newpict[i+(kernel.shape[0]//2)][j+(kernel.shape[1]//2)]=convolve(kernel, pictpart)
    return newpict

```
@Pyodide.eval

***************************************************

### Filter masks for edge detection - theory

--{{0}}--
Edge detection works with gradients in the image. That means you have filter masks that give you a gradient in a certain direction. The higher the distances between two color values are, the stronger is the gradient and the better visible will it be after filtering.

For example:
Gradient in x direction:
| a | b | c |
| a | b | c |
| a | b | c |

> **Remark:** a, b and c are numbers (float or integer is common).

Usually, the following applies:

$ \sum_{a,b,c} = 1$

  {{1}}
***************************************************

> **Remark:** a,b and c may change through a row or column. See in the real filter masks later.

***************************************************

  {{2}}
***************************************************

Other directions:

... gradient in y direction
| a | a | a |
| b | b | b |
| c | c | c |

... gradient in diagonal direction
| a | a | b |
| a | b | c |
| b | c | c |

***************************************************

### Filter masks for edge detection

In the following you will find the most common filter masks implemented:

* Prewitt -> 3x3
+ Sobel -> 3x3
+ Laplace (directional) -> 3x3
- Roberts -> 2x2

#### Prewitt filter masks

```python
prewitt_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype = float)
prewitt_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype = float)

prewitt_45 = np.array([[0,1,1],[-1,0,1],[-1,-1,0]], dtype = float)
prewitt_135 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]], dtype = float)

```
@Pyodide.eval

#### Sobel filter masks

```python
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype = float)
sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype = float)

sobel_45 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]], dtype = float)
sobel_135 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]], dtype = float)

```
@Pyodide.eval

#### Laplace filter masks directional

```python
laplace_x = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype = float)
laplace_y = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype = float)

laplace_45 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], dtype = float)
laplace_135 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], dtype = float)

```
@Pyodide.eval

#### Roberts filter masks directional

```python
roberts_45 = np.array([[0,-1],[1,0]], dtype = float)
roberts_135 = np.array([[-1,0],[0,1]], dtype = float)

```
@Pyodide.eval

### Results of Prewitt filter masks

First compute the new pictures:

```python
img_prewitt_x = filter_image(gray, prewitt_x)
img_prewitt_y = filter_image(gray, prewitt_y)
img_prewitt_45 = filter_image(gray, prewitt_45)
img_prewitt_135 = filter_image(gray, prewitt_135)

```
@Pyodide.eval

  {{1}}
***************************************************

--{{1}}--
Here you see in which direction every filter works.

Plotting:

```python
cmap = plt.get_cmap('gray')

fig, ax = plt.subplots(figsize = (8,8))
plt.suptitle('Prewitt filter operators')
sub1 = plt.subplot(2, 2, 1)
sub1.imshow(img_prewitt_x, cmap = cmap)
sub1.set_title('x direction')
sub2 = plt.subplot(2, 2, 2)
sub2.imshow(img_prewitt_y, cmap = cmap)
sub2.set_title('y direction')
sub3 = plt.subplot(2, 2, 3)
sub3.imshow(img_prewitt_45, cmap = cmap)
sub3.set_title('1st diagonal')
sub4 = plt.subplot(2, 2, 4)
sub4.imshow(img_prewitt_135, cmap = cmap)
sub4.set_title('2nd diagonal')
plt.show()

plot(fig)

```
@Pyodide.eval

***************************************************

### Results of Roberts filter mask

Computations:

```python
img_roberts_45 = filter_image(gray, roberts_45)
img_roberts_135 = filter_image(gray, roberts_135)

```
@Pyodide.eval

  {{1}}
***************************************************

--{{1}}--
This filter mask is a two cross two matrix. This shape results in narrower edges, as you see.

Plotting:

```python
cmap = plt.get_cmap('gray')

fig, ax = plt.subplots()
plt.suptitle('Roberts')
sub1 = plt.subplot(2, 2, 1)
sub1.imshow(img_roberts_45, cmap = cmap)
sub1.set_title('1st diagonal')
sub2 = plt.subplot(2, 2, 2)
sub2.imshow(img_roberts_135, cmap = cmap)
sub2.set_title('2nd diagonal')
plt.show()

plot(fig)

```
@Pyodide.eval

***************************************************

### Compared results of Prewitt and Sobel

Computations:

```python
img_sobel_x = filter_image(gray, sobel_x)
img_sobel_y = filter_image(gray, sobel_y)
img_sobel_45 = filter_image(gray, sobel_45)
img_sobel_135 = filter_image(gray, sobel_135)

```
@Pyodide.eval

{{1}}
***************************************************

--{{1}}--
The Sobel gradient is a little stronger than the Prewitt gradient. That is caused by the higher absolute values in the filter mask.

```python
cmap = plt.get_cmap('gray')

fig, ax = plt.subplots()
plt.suptitle('Compare Prewitt and Sobel')
sub1 = plt.subplot(2, 2, 1)
sub1.imshow(img_prewitt_45, cmap = cmap)
sub1.set_title('Prewitt')
sub2 = plt.subplot(2, 2, 2)
sub2.imshow(img_sobel_45, cmap = cmap)
sub2.set_title('Sobel')
plt.show()

plot(fig)

```
@Pyodide.eval

***************************************************

### Compared results of Prewitt, Sobel and Laplace

Compute the other pictures:

```python
img_laplace_x = filter_image(gray, laplace_x)
img_laplace_y = filter_image(gray, laplace_y)
img_laplace_45 = filter_image(gray, laplace_45)
img_laplace_135 = filter_image(gray, laplace_135)

```
@Pyodide.eval

  {{1}}
***************************************************

--{{1}}--
Here you get an overview about the results of all three filter types. Prewitt and Sobel do nearly the same, Laplace is a bit different. That is, why we will go on with point detection.

```python

imgs = [img_prewitt_x,
        img_prewitt_y,
        img_prewitt_45,
        img_prewitt_135,
        img_sobel_x,
        img_sobel_y,
        img_sobel_45,
        img_sobel_135,
        img_laplace_x,
        img_laplace_y,
        img_laplace_45,
        img_laplace_135
        ]
titles = ["Prewitt x direction",
          "Prewitt y direction",
          "Prewitt 1st diagonal",
          "Prewitt 2nd diagonal",
          "Sobel x direction",
          "Sobel y direction",
          "Sobel 1st diagonal",
          "Sobel 2nd diagonal",
          "Laplace x direction",
          "Laplace y direction",
          "Laplace 1st diagonal",
          "Laplace 2nd diagonal"
          ]

cmap = plt.get_cmap('gray')
fig, ax = plt.subplots(figsize = (12,12))
plt.suptitle('Filter operators')
i = 0
for elem in imgs:
    i += 1
    subi = plt.subplot(4, 4, i)
    subi.imshow(elem, cmap = cmap)
    subi.set_title(titles[i-1])
plt.show()

plot(fig)

```
@Pyodide.eval

***************************************************

## Point detection

point detection = edge detection in every direction

--{{0}}--
Point detection is similar to edge detection in every direction. First we will take one of the filters above and try that out.

### Point detection with Sobel in several directions

--{{0}}--
This method works, but it is laborious. It is more efficient to design a filter mask which does point detection in one step. For that we go on with the Laplacian operator.

```python
img_sobel_2 = 0.5 * (img_sobel_x + img_sobel_y)
img_sobel_4 = 0.25 * (img_sobel_x + img_sobel_y + img_sobel_45 + img_sobel_135)

cmap = plt.get_cmap('gray')
fig, ax = plt.subplots(figsize = (8,8))
plt.suptitle('Sobel filter for point detection')
sub1 = plt.subplot(2, 2, 1)
sub1.imshow(img_sobel_2, cmap = cmap)
sub1.set_title('2 directions')
sub2 = plt.subplot(2, 2, 2)
sub2.imshow(img_sobel_4, cmap = cmap)
sub2.set_title('4 directions')
plt.show()

plot(fig)

```
@Pyodide.eval

### Point detection with Laplace

```python
laplace = np.array([[1,1,1],[1,-8,1],[1,1,1]], dtype = float)

img_laplace = filter_image(gray, laplace)

fig, ax = plt.subplots()
plt.imshow(img_laplace, cmap = cmap)
plt.show()

plot(fig)

```
@Pyodide.eval

## Some additional thoughts

Edge detection is useful for seeing shapes in CT images, for example, but also interesting in any other picture.

Is is easily possible to increase or decrease strength of the gradient or change to a certain direction that is interesting.

Is is also possible to change the size of the filter mask, that will give you only wider edges, not the smallest ones.

You can also take the edges and combine them with any result of filtering. For example, you first take the edges and only blur them or the other way around: You take the edges, blur the rest and add the original edges back onto the picture.

> There are many opportunities to use image filters and edge detection. Just try out, if you are interested! 

