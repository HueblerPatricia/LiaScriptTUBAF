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

$I \approx \dfrac{b-a}{n}\cdot \sum_{n=0}^N \alpha_{j}^{(n)} f(a+jh)$

What do the variables mean?
| variable           | meaning                                |
|--------------------|----------------------------------------|
| I                  | integral                               |
| a                  | lower integration boundary             |
| b                  | upper integration boundary             |
| n                  | number of subdivisions of the interval |
| $\alpha_{j}^{(n)}$ | Newton-Cotes-weights (see below)       |
| f                  | given function                         |
| h                  | step size: $h = \dfrac{b-a}{n}$        |
| j                  | number with $ (j = 0, 1, ..., n) $     |

Newton-Cotes-weights
| n   | name             | nodes in x | weights                                                                              |
|-----|------------------|------------|--------------------------------------------------------------------------------------|
| 1   | trapezoidal rule | 0 1        | $\dfrac{1}{2}$ $\dfrac{1}{2}$                                                        |
| 2   | Simpson-rule     | 0 $\dfrac{1}{2}$  1 | $\dfrac{1}{3}$ $\dfrac{4}{3}$ $\dfrac{1}{3}$                                |
| 3   | 3/8-rule         | 0 $\dfrac{1}{3}$  $\dfrac{2}{3}$  1 | $\dfrac{3}{8}$ $\dfrac{9}{8}$ $\dfrac{9}{8}$ $\dfrac{3}{8}$ |
| ... |                  |                                                                                                   |

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

## The trapezoidal rule in Python

--{{0}}--
How to check, whether our implementation will be correct?
First, take a function with a known integral. For example the integral from zero to one over the derivative of the arctangent.

$\int_{0}^{1} \dfrac{1}{1+x^2} \,dx = arctan(1) - arctan(0) = \dfrac{1}{4} \pi$

Some important functions:

```python
def TrapeziumSurface( a,c,h ):
    A = 0.0
    A = 0.5 * ( a + c ) * h
    return A

def f(x):
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

## Errors depending on subdivisions
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

## Some additional words

It is possible to calculate the maximum error of any Newton-Cotes formula depending on step size and the function to be integrated.

Feel free to modify this script, try out the other Newton-Cotes-formulas or your own ideas!
