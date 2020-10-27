<!--

author:   Patricia Huebler

email:    Patricia.Huebler@student.tu-freiberg.de

language: en

import:  https://github.com/LiaTemplates/Pyodide
-->

# The basics of numerical integration in Python 3
See on LiaScript:

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/NumericalIntegration.md

literature:
* Meister, Andreas u. Sonar, Thomas: Numerik. Eine lebendige und gut verständliche Einführung mit vielen Beispielen. Springer-Verlag GmbH Deutschland, 2019
+ https://www.mathi.uni-heidelberg.de/~thaeter/anasem08/Isenhardt.pdf
- https://www.math.uni-hamburg.de/teaching/export/tuhh/cm/a2/07/vorl12_ana.pdf

See Python documentation if there are any questions about the given implementation.


## What is the problem?
Our goal is to find an approximation for some integral like the following:

$I = \int_{a}^{b} f(x) \,dx$

To solve this problem we need the antiderivative of the given function:

$F(x) = \int_{t_{0}}^{t} f(s) \,ds$

In real situations we can't compute this antiderivative.
So the only opportunity is to calculate the integral numerically.

## Newton-Cotes-formulas
The Newton-Cotes-formulas are a common and easy possibility to solve our problem numerically:

$I \approx Q_{n+1}[f] = \dfrac{b-a}{n}\cdot \sum_{i=0}^n \alpha_{i} f(x_{i}) = h\cdot \sum_{i=0}^n \alpha_{i} f(x_{i})$

What do the variables mean?
| variable           | meaning                                |
|--------------------|----------------------------------------|
| $I$                | integral                               |
| $Q_{n+1}[f]$       | Newton-Cotes-rule                      |
| $a$                | lower integration boundary             |
| $b$                | upper integration boundary             |
| $n$                | number of subdivisions of the interval |
| $\alpha_{i}$       | Newton-Cotes-weights (see below)       |
| $f$                | given function                         |
| $h$                | step size: $h = \dfrac{b-a}{n}$        |
| $i$                | number with $ (i = 0, 1, ..., n) $     |

Newton-Cotes-weights
| n   | name             | nodes in x | weights                                                                              |
|-----|------------------|------------|--------------------------------------------------------------------------------------|
| 1   | trapezoidal rule | 0 1        | $\dfrac{1}{2}$ $\dfrac{1}{2}$                                                        |
| 2   | Simpson-rule     | 0 $\dfrac{1}{2}$  1 | $\dfrac{1}{3}$ $\dfrac{4}{3}$ $\dfrac{1}{3}$                                |
| 3   | 3/8-rule         | 0 $\dfrac{1}{3}$  $\dfrac{2}{3}$  1 | $\dfrac{3}{8}$ $\dfrac{9}{8}$ $\dfrac{9}{8}$ $\dfrac{3}{8}$ |
| ... |                  |                                                                                                   |

> **Remark:** Already implemented in SciPy. But if you really want to understand, what the formulas do, do it yourself!

(source: Meister, Andreas u. Sonar, Thomas: Numerik. Eine lebendige und gut verständliche Einführung mit vielen Beispielen. Springer-Verlag GmbH Deutschland, 2019)

## The trapezoidal rule - first some preparation
Some Python modules to work with ...

```python
import numpy as np
import matplotlib.pyplot as plt
```
@Pyodide.eval

### Some function to be integrated

--{{0}}--
This plot shows the integral to be calculated as the surface below the graph.

```python
x1 = np.linspace( 0, 1, 100 ) # x values
f1 = -(x1-0.5)**2 + .25       # y values by any function

fig, ax = plt.subplots()
plt.plot(x1,f1)
plt.fill_between(x1, f1, color="b", alpha=.1)
plt.show()
plot(fig)

```
@Pyodide.eval

### Integration with one trapezium

```python
x1 = np.linspace( 0, 1, 100 )
f1 = -(x1-0.5)**2 + .25

end = len(x1)-1

# arrays of nodes in x and their accompanying values in y
x_values = [x1[0], x1[end]]
y_values = [f1[0], f1[end]]

fig, ax = plt.subplots()
plt.plot(x1,f1)
plt.plot(x_values, y_values)
plt.fill_between(x_values, y_values, color="r", alpha=.1)
plt.show()
plot(fig)

```
@Pyodide.eval

--{{0}}--
As you see after executing, you see nothing. To integrate we take the starting and ending point of the function in the interval, join them by a straight line and calculate the surface of the trapezium under the red line. But in this case starting point and ending point both are zero. So let's take our interval, subdivide it and see what happens. See on the next slide.

### Integration with two trapezia

```python
x1 = np.linspace( 0, 1, 100 )
f1 = -(x1-0.5)**2 + .25

end = len(x1)-1
middle = 50

x_values = [x1[0], x1[middle], x1[end]]
y_values = [f1[0], f1[middle], f1[end]]

fig, ax = plt.subplots()
plt.plot(x1,f1)
plt.plot(x_values, y_values)
plt.axvline(x1[50], ymin=0.05, ymax=.95, color="r")
plt.fill_between(x_values, y_values, color="r", alpha=.1)
plt.show()
plot(fig)

```
@Pyodide.eval

--{{0}}--
Now we do have a real surface to calculate, but you see the difference between our trapezia and the real graph is quite big. So do another subdivision.

Since we do have more than one trapezium and calculate our integral as sum over all trapezia, we call the process not only "trapezium rule", but "compound trapezium rule".

### Integration with more trapezia

```python
x1 = np.linspace( 0, 1, 100 )
f1 = -(x1-0.5)**2 + .25

end = len(x1)-1

x_values = [x1[0], x1[25], x1[50], x1[75], x1[end]]
y_values = [f1[0], f1[25], f1[50], f1[75], f1[end]]

fig, ax = plt.subplots()
plt.plot(x1,f1)
plt.plot(x_values, y_values)
plt.fill_between(x_values, y_values, color="r", alpha=.1)

plt.axvline(x1[25], ymin=0.05, ymax=.73, color="r")
plt.axvline(x1[50], ymin=0.05, ymax=.95, color="r")
plt.axvline(x1[75], ymin=0.05, ymax=.71, color="r")

plt.show()
plot(fig)

```
@Pyodide.eval

--{{0}}--
You see, the more subdivisions we perform, the better the integral becomes. That means numerical integration by using the trapezoidal rule. And now let's have a look at a possible implementation.

## The trapezoidal rule in Python based on the graphical imagination
--{{0}}--
How to check, whether our implementation will be correct?
First, take a function with a known integral. For example the integral from zero to one over the derivative of the arctangent.

$\Large\int_{0}^{1} \normalsize\dfrac{1}{1+x^2} \,dx = arctan(1) - arctan(0) = \dfrac{1}{4} \pi$

Some important functions:

```python
def TrapeziumSurface( a,c,h ):
    '''Gives the surface of a trapezium

    Parameters
    ----------
    a : number
        length of one of the parallel sides
    c : number
        length of the side parallel to a
    h : number
        distance between a and c

    Returns
    -------
    float
        the surface of the trapezium defined by the parameters

    '''
    A = 0.0
    A = 0.5 * ( a + c ) * h
    return A

def f(x):
    '''Evaluates the formula inside

    Parameters
    ----------
    x : number
        your x values

    Returns
    -------
    float
        calculates y for x from the given formula

    '''
    y = 1 / (1 + x*x)
    return y  
    
```
@Pyodide.eval

```python
#boundaries
a = 0.0
b = 1.0
x = a
dx = 0.001
ya = 0.0
yb = 0.0
trapezium = 0.0
trapezium_sum = 0.0
pi_comp = 0.0
error = 0.0
while (x <= 1):
    ya = f(x)
    x = x + dx
    yb = f(x)
    trapezium = TrapeziumSurface(ya, yb, dx)
    trapezium_sum = trapezium + trapezium_sum

pi_comp = trapezium_sum * 4

print ("Pi is around: ", pi_comp)
print ("The error is: ", (np.pi - pi_comp))

```
@Pyodide.eval

Try some different dx or boundaries to get a feeling of how the process works! Or have a look at the errors as function of the number of subdivisions on the next slide...

### Errors depending on subdivisions
```python
#boundaries:
a = 0.0
b = 1.0

subdivisions = list()
errors = list()

for n in range(1,100):
    x = a
    dx = (b-a)/n
    ya = 0.0
    yb = 0.0
    trapezium = 0.0
    trapezium_sum = 0.0
    pi_comp = 0.0
    error = 0.0

    while (x <= b):
        ya = f(x)
        x = x + dx
        yb = f(x)
        trapezium = TrapeziumSurface(ya, yb, dx)
        trapezium_sum = trapezium + trapezium_sum

    pi_comp = trapezium_sum * 4
    error = np.pi - pi_comp
    subdivisions.append(n)
    errors.append(error)

fig, ax = plt.subplots()
plt.plot(subdivisions,errors)
ax.set_xlabel("number of subdivisions")
ax.set_ylabel("error")
plt.show()
plot(fig)

```
@Pyodide.eval

Theoretically this should be a smooth graph. The peaks are a result of the smallest printable number on a computer. You should always think about this effect, because it is very hard or even impossible to get rid of it.

## The trapezoidal rule in Python based on formulas

--{{0}}--
So now it is clear what we want to do. But there is a much shorter way.

Formula for the trapezoidal rule:

$F_{N}(f) = \dfrac{\Delta x}{2}\cdot \Large\sum_{i=1}^{N}\normalsize (f(x_{i}) + f(x_{i-1}))$

And now we do implement exactly the formula above.

```python
def trapz(f,a,b,N=50):
    '''Approximate the integral of f(x) from a to b by the trapezoid rule.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : integer
        Number of subintervals of [a,b]; 50 by default

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using the
        trapezoid rule with N subintervals of equal length.
    '''
    x = np.linspace(a,b,N+1) # N+1 points make N subintervals
    y = f(x)
    y_right = y[1:] # right endpoints
    y_left = y[:-1] # left endpoints
    dx = (b - a)/N
    T = (dx/2) * np.sum(y_right + y_left)
    return T
    
```
@Pyodide.eval

And now let's try it out:

```python
result = trapz(np.sin,0,np.pi/2,1000)
print(result)
```
@Pyodide.eval

(source for formula and source code: https://www.math.ubc.ca/~pwalls/math-python/integration/trapezoid-rule/)

> **Remark:** ready implemented in scipy.integrate.trapz (see Python documentation for details)

## Simpson's rule
And when you get how the trapezoidal rule works, you can try Simpson's rule.

Formula for Simpson's rule:

$S_{N}(f) = \dfrac{\Delta x}{3}\cdot \Large\sum_{i=1}^{N/2}\normalsize (f(x_{2i-2}) + 4f(x_{2i-1}) + f(x_{2i}))$

```python
def simps(f,a,b,N=50):
    '''Approximate the integral of f(x) from a to b by Simpson's rule.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]; by default 50

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.
    '''
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b-a)/N
    x = np.linspace(a,b,N+1)
    y = f(x)
    S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
    return S
    
```
@Pyodide.eval

A test:

```python
result = simps(np.sin,0,np.pi/2,100)
print(result)
```
@Pyodide.eval

(source for formula and cource code: https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/)

> **Remark:** ready implemented in "scipy.integrate.simps" (see Python documentation for details)

## Some additional words

It is possible to calculate the maximum error of any Newton-Cotes formula depending on step size and the function to be integrated.

Feel free to modify this script, try out the other Newton-Cotes-formulas or your own ideas!
