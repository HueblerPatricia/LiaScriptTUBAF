<!--

author:   Patricia Huebler

email:    Patricia.Huebler@student.tu-freiberg.de

language: en

narrator: UK English Female

import:  https://github.com/LiaTemplates/Pyodide
-->

# Modelling seismic travel time curves of CSG in Python 3

See on LiaScript:

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/TraveltimeCurves.md

> **Remark:** This script deals with P-waves and reflection seismics.

source for this chapter: for example Modul "Seismics I, part II; TU Bergakademie Freiberg, Freiberg; winter semester 2020/2021"

## Import Python modules

```python
import numpy as np
import matplotlib.pyplot as plt

```
@Pyodide.eval

What are the modules for?
| module            | content                                    |
|-------------------|--------------------------------------------|
| NumPy             | work with arrays, matrices, vectors etc.   |
| Matplotlib.Pyplot | plotting images and referred settings      |

## Basics I - The underground model

--{{0}}--
This plot shows what we want to imagine later for programming. We have two layers parted by a defined reflector.

```python
xmodel = np.linspace( 0, 3000, 3000, endpoint = True )
zmodel = np.linspace( 0, 3000, 3000, endpoint = True )
for i in range( 0, len(xmodel) ):
    if ( i < 1000 ):
        zmodel[i] = 1500
    elif ( i < 2000 ):
        zmodel[i] = -0.5 * xmodel[i] + 2000
    else:
        zmodel[i] = 1000

fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( xmodel, zmodel )
plt.ylim(2000,0)
plt.xlim(0,3000)
plt.ylabel("z [m]")
plt.xlabel("x [m]")
plt.title("Model of the underground")
plt.text(550, 400, 'Velocity in upper layer: 2000 m/s', style='italic',
        bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
plt.annotate('diffraction point 1', xy=(1000, 1500), xytext=(500, 1250),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('diffraction point 2', xy=(2000, 1000), xytext=(1500, 750),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.text(350, 1600, 'part I')
plt.text(1400, 1350, 'part II')
plt.text(2350, 1100, 'part III')
plt.show()

plot(fig)

```
@Pyodide.eval

  {{1}}
**************************************************
What variables do we know from that plot?

```python
h1 = 1500       # depth left of 1st diffraction point
h2 = 1000       # depth right of 2nd diffraction point
x_diff1 = 1000  # x position of 1st diffraction point
x_diff2 = 2000  # x position of 2nd diffraction point
v = 2000        # seismic velocity in upper layer
refla = 1000    # left end of reflector in x
reflb = 2000    # right end of reflector in x
reflh = -500    # difference in depth of reflector
reflm = (reflh) / (reflb - refla) # incline of reflector
z_refl_0 = 2000 # depth of reflector in x = 0

```
@Pyodide.eval

**************************************************

  {{2}}
**************************************************

> **Remark:** Python works with radians instead of degree, so let's write a converting function.

```python
def rad(alpha):
    '''
    Converts degree to radians

    Parameter:
        alpha(number): angle in degree

    Returns:
        beta(number): angle in radians
    '''
    beta = (np.pi * alpha) / 180.0
    return beta

```
@Pyodide.eval

**************************************************

  {{3}}
**************************************************

What do we need further?

```python
phi = -26.56505118 # = atan(reflm)
phir = rad(phi)

```
@Pyodide.eval

**************************************************

## The setting of source and geophones:

For this first example we use a setting called "CSG" (common shot gather). That means, our source stays at the same point for the acquisition.

--{{0}}--
For our first example we have our seismic source at position x is equal zero. Our geophones are located from x equal zero to x equal three thousand.

```python
x_shot = 0
```
@Pyodide.eval

## First shot gather - shot at x = 0m
What parts of the underground do we meet?

> **Remark:** Reflection: $\alpha$ = $\alpha '$ !

* part I -> ???
+ diffraction point 1 -> ???
+ part II -> ???
+ diffraction point 2 -> ???
- part III -> ???



  {{1}}
**************************************************

**part I: Let's see!**

```python
demox1 = [0, 500, 1000]
demoy1 = [0, 1500, 0]

fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( xmodel, zmodel )
plt.plot( demox1, demoy1 )
plt.ylim(2000,0)
plt.xlim(0,3000)
plt.ylabel("z [m]")
plt.xlabel("x [m]")
plt.title("How do we see the model's part I?")
plt.annotate('source', xy=(0, 0), xytext=(200, 250),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('example geophone', xy=(1000, 0), xytext=(1120, 250),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

plot(fig)

```
@Pyodide.eval

**************************************************

  {{2}}
**************************************************

**diffraction point 1: Let's see!**

--{{2}}--
Diffraction points are points that reflect waves in every direction. So for example you meet the following three points with diffraction point one.

```python
demox1 = [0, 1000, 500 ]
demoy1 = [0, 1500, 0 ]
demox2 = [0, 1000, 1500 ]
demox3 = [0, 1000, 3000 ]

fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( xmodel, zmodel )
plt.plot( demox1, demoy1, color = 'red' )
plt.plot( demox2, demoy1, color = 'red' )
plt.plot( demox3, demoy1, color = 'red' )
plt.ylim(2000,0)
plt.xlim(0,3000)
plt.ylabel("z [m]")
plt.xlabel("x [m]")
plt.title("How do we see diffraction point 1?")
plt.annotate('source', xy=(0, 0), xytext=(200, 250),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

plot(fig)

```
@Pyodide.eval

**************************************************

  {{3}}
**************************************************

**part II: Let's see!**

--{{3}}--
We see part two, but our seismic wave will meet the reflector in a different angle.

```python
demox1 = [0, 1500, 1435 ]
demoy1 = [0, 1250, 0 ]

fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( xmodel, zmodel )
plt.plot( demox1, demoy1, color = 'red' )
plt.ylim(2000,0)
plt.xlim(0,3000)
plt.ylabel("z [m]")
plt.xlabel("x [m]")
plt.title("How do we see part II?")
plt.annotate('source', xy=(0, 0), xytext=(200, 250),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

plot(fig)

```
@Pyodide.eval

**************************************************

  {{4}}
**************************************************

**diffraction point 2: Let's see!**

--{{4}}--
That is to imagine as for diffraction point one.

```python
demox1 = [0, 2000, 500 ]
demoy1 = [0, 1000, 0 ]
demox2 = [0, 2000, 1500 ]
demox3 = [0, 2000, 3000 ]

fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( xmodel, zmodel )
plt.plot( demox1, demoy1, color = 'red' )
plt.plot( demox2, demoy1, color = 'red' )
plt.plot( demox3, demoy1, color = 'red' )
plt.ylim(2000,0)
plt.xlim(0,3000)
plt.ylabel("z [m]")
plt.xlabel("x [m]")
plt.title("How do we see diffraction point 2?")
plt.annotate('source', xy=(0, 0), xytext=(200, 250),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

plot(fig)

```
@Pyodide.eval

**************************************************

  {{5}}
**************************************************

**part III: Let's (not) see!**

--{{5}}--
Because of the reflection formula alpha is equal alpha reflected, we don't see part three of our model.

```python
demox1 = [0, 2500, 5000]
demoy1 = [0, 1000, 0]

fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( xmodel, zmodel )
plt.plot( demox1, demoy1 )
plt.ylim(2000,0)
plt.xlim(0,3000)
plt.ylabel("z [m]")
plt.xlabel("x [m]")
plt.title("How do(n't) we see the model's part III?")
plt.annotate('source', xy=(0, 0), xytext=(200, 250),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
plot(fig)

```
@Pyodide.eval

**************************************************

## Formula for travel time

Most important formula:

> $v = \dfrac{s}{t} \Leftrightarrow t = \dfrac{s}{v}$

With this formula and some drawing and imagination you will get the rest.

## Functions for all we have seen

> **Remark:** Drawing on a sheet of paper is much easier than drawing via source code. So I recommend for you to take the pictures above step by step and try to understand what the following functions do.

For the following calculations we will need some functions from the Python modules:

```python
from math import atan
from scipy import sqrt, tan, cos, sin, pi


```
@Pyodide.eval

### Part I - horizontal reflector

```python

def csg(x, x_shot, h, v):

    '''
    travel times for common shot gather above horizontal reflector

    Parameters:
        x(array 1D)   : x positions of geophones
        x_shot(number): x position of source
        h(number)     : depth between surface and reflector
        v(number)     : velocity in upper layer

    Returns:
        t(array 1D): travel times
    '''

    t = sqrt(((4 * h**2 + (x - x_shot)**2) / v**2))
    return t

x1 = np.linspace(0, 2000, 2000, endpoint = True) # define our geophone positions
t1 = csg(x1, x_shot, h1, v)

```
@Pyodide.eval


{{1}}
**************************************************

```python
fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( x1, t1, color = 'cornflowerblue', label = "part I" )
plt.legend()
plt.ylim(2.5,0)
plt.xlim(0,3000)
plt.ylabel("t [s]")
plt.xlabel("x [m]")
plt.title("Seismic travel times")
plt.show()

plot(fig)

```
@Pyodide.eval

**************************************************

### Diffraction point 1

```python

def diffraction(x, x_shot, x_diff, h_diff, v):
    '''
    travel times for diffraction point

    Parameters:
        x(array 1D)   : x positions of geophones
        x_shot(number): x position of source
        x_diff(number): x position of diffraction point
        h_diff(number): depth of diffraction point
        v(number)     : velocity in upper layer

    Returns:
        t(array 1D): travel times
    '''

    abs1_x = abs(x_shot - x_diff)
    abs_y = abs(h_diff)
    l1 = sqrt(abs1_x**2 + abs_y**2)
    abs_x = abs(x - x_diff)
    l2 = sqrt(abs_x**2 + abs_y**2)

    t = (l1 + l2) / v
    return t

x2 = np.linspace(0, 3000, 3000, endpoint = True)
t2 = diffraction(x2, x_shot, x_diff1, h1, v)

```
@Pyodide.eval


{{1}}
**************************************************

```python
fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( x1, t1, color = 'cornflowerblue', label = "part I" )
plt.plot( x2, t2, color = 'coral', label = "diffraction point 1" )
plt.legend()
plt.ylim(2.5,0)
plt.xlim(0,3000)
plt.ylabel("t [s]")
plt.xlabel("x [m]")
plt.title("Seismic travel times")
plt.show()

plot(fig)
```
@Pyodide.eval

**************************************************


### Diffraction point 2

```python
x4 = np.linspace(0, 3000, 3000, endpoint = True)
t4 = diffraction(x4, x_shot, x_diff2, h2, v)

```
@Pyodide.eval


{{1}}
**************************************************

```python
fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( x1, t1, color = 'cornflowerblue', label = "part I" )
plt.plot( x2, t2, color = 'coral', label = "diffraction point 1" )
plt.plot( x4, t4, color = 'peru', label = "diffraction point 2" )
plt.legend()
plt.ylim(2.5,0)
plt.xlim(0,3000)
plt.ylabel("t [s]")
plt.xlabel("x [m]")
plt.title("Seismic travel times")
plt.show()

plot(fig)
```
@Pyodide.eval

**************************************************

### Part II - reflector with angle

```python

def reflector_angle(h, v, xl, xr, phir):

    '''
    travel times for reflector with angle and common shot gather

    Parameter:
        h(number)   : vertical depth of reflector at x = shot point
        v(number)   : velocity in upper layer
        xl(number)  : left end of reflector in x
        xr(number)  : right end of reflector in x
        phir(number): angle of reflector in radians

    Returns:
        x(array 1D): positions of geophones
        t(array 1D): travel times
    '''

    m = tan(phir)
    xstart = 440
    xend = 2200
    x = np.linspace(xstart, xend, int(xend - xstart + 0.5), endpoint = True)
    h_angle = (m*x + h) * cos(phir)
    temp = sqrt( x**2 + 4*(h_angle**2) - (4 * h_angle * x * sin(phir)))
    t = temp / v
    return x,t


x3, t3 = reflector_angle(z_refl_0, v, refla, reflb, phir)

```
@Pyodide.eval


{{1}}
**************************************************

```python
fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( x1, t1, color = 'cornflowerblue', label = "part I" )
plt.plot( x3, t3, color = 'mediumblue', label = "part II" )
plt.plot( x2, t2, color = 'coral', label = "diffraction point 1" )
plt.plot( x4, t4, color = 'peru', label = "diffraction point 2" )
plt.legend()
plt.ylim(2.5,0)
plt.xlim(0,3000)
plt.ylabel("t [s]")
plt.xlabel("x [m]")
plt.title("Seismic travel times")
plt.show()

plot(fig)
```
@Pyodide.eval

**************************************************

### Travel time diagram including a direct wave

```python

def direct(x, x_shot, v):

    '''
    travel times for direct wave

    Parameters:
        x(array 1D)     : x positions of geophones
        x_schuss(number): x position of source
        v(number)       : velocity in upper layer

    Returns:
        t(array 1D): travel times
    '''

    abs_x = abs(x_shot - x)
    t = abs_x / v
    return t


xdirect = np.linspace(0, 3000, 3000, endpoint = True)
tdirect = direct(xdirect, x_shot, v)

```
@Pyodide.eval


{{1}}
**************************************************

```python
fig, ax = plt.subplots()
plt.gca().invert_yaxis()
plt.plot( xdirect, tdirect, color = 'gray', label = "direct wave" )
plt.plot( x1, t1, color = 'cornflowerblue', label = "part I" )
plt.plot( x3, t3, color = 'mediumblue', label = "part II" )
plt.plot( x2, t2, color = 'coral', label = "diffraction point 1" )
plt.plot( x4, t4, color = 'peru', label = "diffraction point 2" )
plt.legend()
plt.ylim(2.5,0)
plt.xlim(0,3000)
plt.ylabel("t [s]")
plt.xlabel("x [m]")
plt.title("Seismic travel times")
plt.show()

plot(fig)
```
@Pyodide.eval

**************************************************

### Other seismic travel times

If the formulas are known to you, it is also easily possible to implement functions for ZOG (zero offset gather) or for CMP (common midpoint gather), for example. Just try out if you are interested!
