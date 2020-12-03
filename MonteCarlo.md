<!--

author:   Patricia Huebler

email:    Patricia.Huebler@student.tu-freiberg.de

language: en

narrator: UK English Female

import:  https://github.com/LiaTemplates/Pyodide
-->

# Approximation for pi using the Monte Carlo integration method and Python 3

literature: https://en.wikipedia.org/wiki/Monte_Carlo_method

See on LiaScript:

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/MonteCarlo.md

## Preparations
Import necessary modules

```python
import numpy as np               
import matplotlib.pyplot as plt  
from scipy import sqrt                   

```
@Pyodide.eval

What are the modules for?
| module            | content                                    |
|-------------------|--------------------------------------------|
| NumPy             | work with arrays, matrices, vectors etc.   |
| Matplotlib.Pyplot | plotting images and referred settings      |
| sqrt from SciPy   | calculations with array-like objects       |
| random            | getting random numbers                     |

## How does Monte Carlo integration work?

--{{0}}--
We have a function to use and the integration limits a and b.

given:
* function f(x) to be integrated (higher dimensions optional)
- integration borders as interval [a;b]


  {{1}}
**********************************************************

--{{1}}--
A and b are the edges of a square that we need. Then we scatter an amount of random points into this square and count, how many of them are located in the surface under the graph. In relation with the entire surface of the square this is the approximation for the integral.

method:
* define a square (or a rectangle) around the graph using a and b as left and right and as lower and upper border
+ create a set of random points in this rectangle or square
+ count points under the graph and above the graph
+ compute probability for a point ending up under the graph
- compare probability with total surface of rectangle $\rightarrow $ that is the integral

**********************************************************

## First functions

A function that calculates f(x) for a circle line.

```python
def circle_func(x):

    '''calculates f(x)

    Parameters
    ----------
    x : number or array-like
        x coordinate(s)

    Returns
    -------
    number or array-like
        y coordinate(s)
    '''

    y = sqrt( r**2 - x**2 )
    return y

```
@Pyodide.eval

{{1}}
**********************************************************

Define the circle's radius and some x and y values for drawing.

```python
r = 1
x1 = np.linspace( 0, r, 100 )
f1 = circle_func(x1)

```
@Pyodide.eval


**********************************************************


{{2}}
**********************************************************

Function to find random points in a square.

```python
def rand_points_square(n,a,b):

    '''gives random points in 2D

    Parameters
    ----------
    n : number
        amount of points
    a , b : numbers
        Interval of integration [a,b]

    Returns
    -------
    2 arrays of 1D
        x and y coordinates of random points
    '''


    np.random.seed()
    list_x = list()
    list_y = list()
    for i in range(0,n):
        x = np.random.random() * b
        y = np.random.random() * b
        list_x.append(x)
        list_y.append(y)
    points_x = np.asarray(list_x)
    points_y = np.asarray(list_y)
    return points_x, points_y

```
@Pyodide.eval


**********************************************************

## An illustration

Define all we need.

```python
a = 0
b = r
n = 1000
points_x, points_y = rand_points_square(n,a,b)

```
@Pyodide.eval

  {{1}}
**********************************************************

--{{1}}--
Here we see how Monte Carlo integration works. We have an amount of randomly scattered points. All that are located in the blue area belong to the surface we want to calculate.

An illustration of the Monte Carlo integration method.

```python
fig, ax = plt.subplots(figsize = (5,5))
plt.plot(x1,f1)
plt.scatter(points_x, points_y, s = 10, color = 'blue', marker = ".")
plt.xlim(0,r)
plt.ylim(0,r)
plt.axis('equal')
plt.fill_between(x1, f1, color="b", alpha=.1)
plt.show()

plot(fig)

```
@Pyodide.eval

**********************************************************

## Next things to do ...

* find out, which point belongs to surface under graph, which does not and count one of them
- calculate surface

## Count points located on surface

A function to do that counting for us.

```python
def counts_under_graph(x,y,func):

    '''counts points under graph

    Parameters
    ----------
    x, y : array-like
        coordinates of points
    func : string
        the function to be considered

    Returns
    -------
    integer
        number of points under graph
    '''

    counts_true = 0

    for i in range(0, len(x)-1):
        y2 = func(x[i])
        distance = y2 - y[i]
        if distance >= 0:
            counts_true += 1
        else:
            counts_true = counts_true
    return counts_true

```
@Pyodide.eval


  {{1}}
**********************************************************

Execute it.

```python
counts = counts_under_graph(points_x, points_y, circle_func)
print(counts)

```
@Pyodide.eval

**********************************************************

## Calculate approximation for surface

$ \dfrac{surface}{entire} = \dfrac{counts}{n} \Leftrightarrow  surface = \dfrac{entire\cdot counts}{n}$

```python
entire = b**2
surface = (entire*counts)/n
print(surface)

```
@Pyodide.eval

## We have the surface - now the approximation for pi!

$ surface\: of\: circle: \qquad A = \pi \cdot r^2 \Leftrightarrow \pi = \dfrac{A}{r^2} $

We have a circle's fourth, so ...

$ \pi = \dfrac{4\cdot surface}{r^2} $



  {{1}}
**********************************************************

--{{1}}--
The error is quite large at the moment. To reduce it we can only increase n. For that we define a function for the whole approximation process in the next step.

```python
pi_approx = (4*surface)/(r**2)
difference = abs(np.pi - pi_approx)

print("Approximation: pi = ", pi_approx)
print("Error: ", difference)

```
@Pyodide.eval


**********************************************************


## Function for approximation

```python
def approx_pi(n,a,b,r):

    '''approximation for pi using the functions above

    Parameters
    ----------
    n : integer
        number of random points
    a,b : numbers
        limits of integration interval
    r : number
        the circle's radius

    Returns
    -------
    float, float
        approximation of pi, It's error
    '''

    points_x, points_y = rand_points_square(n,a,b)
    counts = counts_under_graph(points_x, points_y, circle_func)
    surface = (entire*counts)/n
    pi_approx = (4*surface)/(r**2)
    difference = abs(np.pi - pi_approx)
    return pi_approx, difference

```
@Pyodide.eval

  {{1}}
**********************************************************

--{{1}}--
And now you may try out the approximation for different n and see how the error behaves.

```python
n = 100000

pi_approx, difference = approx_pi(n,a,b,r)

print("Approximation: pi = ", pi_approx)
print("Error: ", difference)

```
@Pyodide.eval

**********************************************************

## Behaviour of the error

--{{0}}--
Let's try out with different amounts of randomly scattered points and see how the error behaves.

```python
n_array = np.linspace(20, 10000, 20)
diff_list = list()
n_list = list()

for i in range(0, len(n_array)-1):
    nvar = int(n_array[i])
    pi_approx, diff = approx_pi(nvar,a,b,r)
    n_list.append(nvar)
    diff_list.append(diff)

n_array = np.asarray(n_list)
diff_array = np.asarray(diff_list)

fig, ax = plt.subplots()
plt.plot(n_array, diff_array)
plt.xlabel("number of random points")
plt.ylabel("error")
plt.show()

plot(fig)

```
@Pyodide.eval

## Conclusion

Monte Carlo integration is a very simple and easily imaginable opportunity for approximating some integrals. But the error depends on a random variable, that means on coincidence.

But nevertheless it is interesting to see how such numerical methods work and to have the opportunity to play around with them a bit. Feel free to do that with this script or in Jupyter Notebook, if you are interested.

link to the concerning Jupyter Notebook: https://github.com/HueblerPatricia/JupyterNotebooksTUBAF/blob/main/MonteCarlo.ipynb
