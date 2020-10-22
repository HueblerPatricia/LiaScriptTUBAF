<!--

author:   Patricia Huebler

email:    Patricia.Huebler@student.tu-freiberg.de

language: en

import:  https://github.com/LiaTemplates/Pyodide
-->

# Python 3 - An introduction to NumPy

```python
import numpy as np
```
@Pyodide.eval

See on LiaScript: 

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/IntroductionToNumpy.md

## What is NumPy?

    --{{0}}--
NumPy means numerical python. It is made to work with multidimensional array
objects, such as matrices and vectors. And there is a collection of operations
for working with these objects. When you use NumPy together with the packages SciPy (which means scientific
python) and Matplotlib, you get a functional opportunity to replace MATLAB.

* NumPy = numerical Python (working with multidimensional array objects)
- NumPy + SciPy + Matplotlib = replacement for MATLAB

> **Remark:** Referring to array objects Python doesn't distinguish between vectors and matrices. They both are just arrays having different dimensions.


## Basics

### Vector creation

```python
# Create a simple vector
a = np.array([1,0,2])
print (a)
a.dtype
```
@Pyodide.eval

=> Data types are chosen automatically except you chose yourself.

    --{{0}}--
The data type is chosen automatically. But sometimes that will not be what you
need, so you can do this yourself. See below.

      {{1}}
```python
# Create another vector
C = np.array([1,2,0], dtype=complex )
print (C)
C.dtype
```
@Pyodide.eval

    --{{1}}--
You can choose between all Python data types, such as integer, float, boolean,
single characters, complex numbers or others (see Python documentation for
this).

### Matrix creation with known entries

If all elements known, use following implementation:

```python
# Create matrix like vector, both are arrays
B = np.array([[1,2,0],[2,0,1],[0,1,2]])
print (B)
B.dtype
```
@Pyodide.eval

> **Remark:** Normally Python does print the arrays with ordered rows and
> columns. This line style is also a little problem of LiaScript.

### Matrix creation with unknown entries

    --{{0}}--
What if you just know the dimension but not all entries? That is also possible!
For example in the following ways.

```python
D = np.zeros((2,3), dtype=np.int16 )
print ("D=\n",D)
E = np.zeros((2,3))
print ("E=\n",E)
```
@Pyodide.eval

> **Remark:** You can also create arrays of higher dimensions if needed.

## Functions evaluation with arrays

### Create x

    --{{0}}--
Let's imagine you have to plot a function f depending on x. To plot that Python needs
points. So first we create an array consisting of x.

```python
#np.arange(startpoint, endpoint, stepsize)
x1 = np.arange( 0, 2.1, 0.1 )
print (x1)

#np.linspace(startpoint, endpoint, number of points in between -> useful for many entries)
from numpy import pi
x2 = np.linspace( 0, 2*pi, 100 )
#print (x2)
```
@Pyodide.eval

### Create y

```python
f1 = np.sin(x2)
print ("f1", f1)

f2 = x1**2
print ("f2", f2)
```
@Pyodide.eval

> **Remark:** You can always check your results by simply printing them.

### Plotting a function

```python
import matplotlib.pyplot as plt
```
@Pyodide.eval

Normally you would use this code:

```python
plt.plot(x2,f1)
plt.show()
```

But that doesn't work in LiaScript, so we do the following:

```python
fig, ax = plt.subplots()
plt.plot(x2,f1)
plt.show()
plot(fig)
```
@Pyodide.eval

```python
fig2, ax = plt.subplots()
plt.plot(x1,f2)
plt.show()
plot(fig2)
```
@Pyodide.eval

## Combining commands

    --{{0}}--
It is also possible to combine different commands. For example you can create a
matrix out of a set of ordered numbers.

```python
F = np.arange(20).reshape(5,4)
print (F)
```
@Pyodide.eval

=> Indeces beginning with zero. Don't want? Try this:

    --{{0}}--
Here you see that Python is indexed beginning with zero. If you don't want this
you have to consider it in the command line. For example change the one above to
the following.

```python
F2 = np.arange(1,21,1).reshape(5,4)
print (F2)
```
@Pyodide.eval

## Various computations with arrays

### Sums of columns and rows

```python
F2.sum(axis=0)    # sum of each column
```
@Pyodide.eval

```python
F2.sum(axis=1)    # sum of each row
```
@Pyodide.eval

### Some mathematical operations with vectors

```python
A = np.arange(3)
print (A)

expA = np.exp(A)
print (expA)

sqrtA = np.sqrt(A)
print (sqrtA)

B = np.array([2., 0., -1.])
AplusB = np.add(A, B)
print (AplusB)

```
@Pyodide.eval

### Some mathematical operations with matrices

Again we take our matrix `F`.

```python
print (F)
```
@Pyodide.eval

#### Iterating over a matrix

    --{{0}}--
Iterating in a multidimensional array is always done with respect to the first
axis.

      {{0}}
```python
for row in F:
    print (row)
```
@Pyodide.eval

    --{{1}}--
If you want to iterate somehow else, it is a bit more complicated. Just be
creative or look it up. Python is that popular that there is nearly always
someone who already found a solution for your problem. For example let's do an
iteration over the columns of the matrix.

      {{1}}
```python
for column in F.T:  # iterating over rows in transposed matrix
    print (column)
```
@Pyodide.eval

> **Remark:** It doesn't make any difference whether you call those indices `row`,
> `column` or just `i`, `j`, `index` or somehow else. They are just indices,
> variables if you want.

#### Matrix multiplication

```python
A = np.array([[1,2],[0,-1]], dtype=np.float)
B = np.array([[3,-2],[1,-1]], dtype=np.float)
AdotB = A.dot(B)    # the "common" matrix product
print (AdotB)
```
@Pyodide.eval

    --{{0}}--
This is just one of many ready implemented functions. You will find a suitable
one for any problem.

There is also a cross product that could be suitable for physics:
`np.cross(A,B)`. Further we also do have `np.tensordot(A,B)` with optional
argument `axis=1`. It doesn't matter what you need, just look into the Python
ducomentation and be sure you will find something useful.

#### Invert a matrix

Aditionally we need another module for this.

```python
import scipy.linalg as la
```
@Pyodide.eval

```python
print (B)
```
@Pyodide.eval

```python
Binv = la.inv(B)
print (Binv)
```
@Pyodide.eval

### Solving linear equation systems

```python
#Do only load modul if you haven't done already.
import scipy.linalg as la
```
@Pyodide.eval

```python
print (AdotB)
```
@Pyodide.eval

```python
vec_b = np.array([1,-2.5], dtype = np.float)
print (vec_b)
```
@Pyodide.eval

```python
x = la.solve(AdotB,vec_b)
print (x)
```
@Pyodide.eval

## Some additional words

There are far more possibilities with NumPy, SciPy, Pandas and other modules. Feel free to check out things in this basic script and have a look into the Python documentation. There is nearly always a solution for any problem.
