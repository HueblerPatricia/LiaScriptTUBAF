<!--

author:   Patricia Huebler

email:    Patricia.Huebler@student.tu-freiberg.de

language: en

narrator: UK English Female

import:  https://github.com/LiaTemplates/Pyodide
-->

# Introduction to digital signal processing in Python 3

literature for this chapter:

* Steven W. Smith: The Scientist's and Engineer's Guide to Digital Signal Processing. Second Edition. California Technical Publishing, USA 1997 - 1999
- http://iowahills.com/


See on LiaScript:

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/DigitalSignalProcessing.md

## Python modules

--{{0}}--
First we bind in some Python modules.

```python
import numpy as np
import scipy as sp
import scipy.signal as sg #does not completely work in LiaScript (yet)
import scipy.fftpack as fp
import matplotlib.pyplot as plt
```
@Pyodide.eval

What are the modules for?
| module            | content                                    |
|-------------------|--------------------------------------------|
| NumPy             | work with arrays, matrices, vectors etc.   |
| SciPy             | mathematics, science, engineering          |
| SciPy.Signal      | signal processing                          |
| SciPy.fftpack     | (fast) fourier transformation              |
| Matplotlib.Pyplot | plotting images and referred settings      |

## The sampling theorem

"If you can exactly reconstruct the analog signal from the samples, you must have done the sampling properly." *- The Scientist's and Engineer's Guide to Digital Signal Processing, page 39*

"The sampling theorem indicates that a continuous signal can be properly sampled, only if it does not contain frequency components above one-half of the sampling rate." *- The Scientist's and Engineer's Guide to Digital Signal Processing, page 40*

Therefor the **Nyquist frequency** or **Nyquist rate** is defined. In *"The Scientist's and Engineer's Guide to Digital Signal Processing"* it is defined as
$f_{Ny} = 0.5\cdot sample rate = \dfrac{1}{2\Delta t}$

> "The digital signal cannot contain frequencies above one-half the sampling rate." - The Scientist's and Engineer's Guide to Digital Signal Processing, page 42 

**In easy words:** Let's assume you have a signal containing frequencies lower or equal 50 Hz. Then you have to choose your sample rate minimum 100 Hz.

Keep that in mind! You will need it often in signal processing although it is not that important for this example chapter.

## Some sample rates to choose from

  --{{0}}--
We will use this sample rate through the whole chapter.

```python
#index      =   0    1    2    3     4    
sampleRates = (250, 300, 400, 500, 1000)

sampleRate = sampleRates[3] #Hz

```
@Pyodide.eval

{{1}}
******************************************************************
Define a time line: How long do you want it to be?

```python
endTime = 1 # seconds

t = np.linspace(0, endTime, endTime*sampleRate, endpoint = True)
x = np.arange(len(t))

```
@Pyodide.eval
******************************************************************

### Example signals

**Sine signal**

--{{0}}--
A sine signal belongs to the most common signals. It is the most important one in acoustics. But we do also know it in seismics for example.

```python
frqSine = 50
amplSine = 1
sgSine = amplSine * np.sin(2 * np.pi * frqSine * x/sampleRate)

```
@Pyodide.eval


{{1}}
******************************************************************

**Square-wave signal**

--{{1}}--
A square-wave signal is a periodic signal that changes between two values. Ideal squares only exist theoretically. They can be created by overlaying many freqencys of a sine signal.

```python
frqSquare = 30
dutySquare = 0.5
amplSquare = 10
sgSquare = amplSquare * sg.square(2 * np.pi * frqSquare * x/sampleRate, dutySquare)

```
@Pyodide.eval

******************************************************************

{{2}}
******************************************************************

**Sweep**

--{{2}}--
A sweep is a signal we know from seismic exploration. It's frequency changes over time from low to high or the other way around. Technically changing from a low to a high frequency is sometimes easier.

```python
sgSweep = sg.chirp(t, f0=10, f1=200, t1=endTime, method='linear')
envelope = 1 * np.sin(2 * np.pi * 0.5 * x/sampleRate)
sgSweep *= envelope

```
@Pyodide.eval

******************************************************************

### Have a look at our signals

```python
fig, ax = plt.subplots(3,1, figsize = (12,12))
ax[0].plot(t, sgSine, label='simulated sine signal', lw = .3)
ax[0].set_title("Sine signal")
ax[0].set_xlabel('time [s]')
ax[0].set_ylabel('amplitude')

ax[1].plot(t, sgSquare, label='simulated square-wave signal', lw = .3)
ax[1].set_title("Square-wave signal")
ax[1].set_xlabel('time [s]')
ax[1].set_ylabel('amplitude')

ax[2].plot(t, sgSweep, label='simulated sweep signal', lw = .3)
ax[2].set_title("Sweep")
ax[2].set_xlabel('time [s]')
ax[2].set_ylabel('amplitude')

plt.subplots_adjust(hspace=0.4)
plt.show()

plot(fig)

```
@Pyodide.eval

> **Remark:** If the sweep looks a bit frayed, that is a sampling problem (see sampling theorem). You may change the sample rate if you want.

## FFT : Fast Fourier Transformation

"Fourier analysis converts a signal from its original domain (often time [...]) to a representation in the frequency domain [...]"
source: https://en.wikipedia.org/wiki/Fast_Fourier_transform

So let's do an FFT for all of our 3 signals!

```python
sin_fft = fp.fft(sgSine)
sin_abs_fft = 2 * np.abs(sin_fft)/(len(sgSine))
sin_fftfreq = fp.fftfreq(len(sin_abs_fft), 1/sampleRate)

sq_fft = fp.fft(sgSquare)
sq_abs_fft = 2 * np.abs(sq_fft)/(len(sgSquare))
sq_fftfreq = fp.fftfreq(len(sq_abs_fft), 1/sampleRate)

sweep_fft = fp.fft(sgSweep)
sweep_abs_fft = 2 * np.abs(sweep_fft)/(len(sgSweep))
sweep_fftfreq = fp.fftfreq(len(sweep_abs_fft), 1/sampleRate)

```
@Pyodide.eval


{{1}}
******************************************************************

**Plots**

```python
fig, ax = plt.subplots(3,1, figsize = (10,10))
i = sin_fftfreq > 0
fny = 0.5 * samplerate

ax[0].plot(sin_fftfreq[i], sin_abs_fft[i], lw = .7)
ax[0].set_title("Spectrum of sine signal")
ax[0].set_xlabel('frequency [Hz]')
ax[0].set_ylabel('amplitude')
ax[0].set_xlim(0,fny)
ax[0].grid()

ax[1].plot(sq_fftfreq[i], sq_abs_fft[i], lw = .7)
ax[1].set_title("Spectrum of square-wave signal")
ax[1].set_xlabel('frequency [Hz]')
ax[1].set_ylabel('amplitude')
ax[1].set_xlim(0,fny)
ax[1].grid()

ax[2].plot(sweep_fftfreq[i], sweep_abs_fft[i], lw = .7)
ax[2].set_title("Spectrum of sweep")
ax[2].set_xlabel('frequency [Hz]')
ax[2].set_ylabel('amplitude')
ax[2].set_xlim(0,fny)
ax[2].grid()

plt.subplots_adjust(hspace=0.4)
plt.show()

plot(fig)

```
@Pyodide.eval

******************************************************************

## Frequency filtering - A simple sine signal with noise

What to do?
* create the signal
+ add noise (in example 50 Hz of an electric cable)
+ convertion to frequency domain (FFT)
- mute the noise by filtering in frequency domain

### Create the time series

```python
sampleRate = sampleRates[3]
endTime = 1   #seconds

frqSine = 20
amplSine = 0.7
amplNoise = 0.3
frqNoiseSine = 50

t = np.linspace(0, endTime, endTime*sampleRate, endpoint = True)
x = np.arange(len(t))
sgSine = amplSine * np.sin(2 * np.pi * frqSine * x/sampleRate)
sgNoise = amplNoise * np.sin(2 * np.pi * frqNoiseSine * x/sampleRate)

sgSum = sgSine + sgNoise

```
@Pyodide.eval


### FFT and plot

```python
sin_fft = fp.fft(sgSine)
sin_abs_fft = 2 * np.abs(sin_fft)/(len(sgSine))
sin_fftfreq = fp.fftfreq(len(sin_abs_fft), 1/sampleRate)

sg_fft = fp.fft(sgSum)
sg_abs_fft = 2 * np.abs(sg_fft)/(len(sgSum))
sg_fftfreq = fp.fftfreq(len(sg_abs_fft), 1/sampleRate)

```
@Pyodide.eval

--{{0}}--
In frequency domain we clearly see two parted frequencies. One is the signal, the other is the noise.

```python
fig = plt.figure(figsize = (8,6))
i = sin_fftfreq > 0
sub1 = fig.add_subplot(2,2,1)
sub1.plot(t, sgSine)
sub1.set_title('The original single sine wave signal')
sub1.set_xlabel('time [s]')
sub1.set_ylabel('amplitude')
sub1.set_ylim(-1.1,1.1)
sub2 = fig.add_subplot(2,2,2)
sub2.plot(t, sgSum)
sub2.set_title('The signal with noise')
sub2.set_xlabel('time [s]')
sub2.set_ylim(-1.1,1.1)
sub3 = fig.add_subplot(2,2,3)
sub3.plot(sin_fftfreq[i], sin_abs_fft[i])
sub3.set_title('The original single sine wave signal')
sub3.set_xlabel('time [s]')
sub3.set_ylabel('amplitude')
sub3.set_xlim(0,100)
sub4 = fig.add_subplot(2,2,4)
sub4.plot(sg_fftfreq[i], sg_abs_fft[i])
sub4.set_title('The signal with noise')
sub4.set_xlabel('time [s]')
sub4.set_xlim(0,100)
plt.subplots_adjust(hspace=0.4)
plt.show()

plot(fig)

```
@Pyodide.eval

### Four common frequency filters

Where the filter amplitude is 1, the amplitudes of the signal stay unchanged, where it is 0, the signal's amplitudes become 0.

```python
lowpass_demo = [[0, 25, 35, 80], [1, 1, 0, 0]]
highpass_demo = [[0, 10, 20, 80], [0, 0, 1, 1]]
bandpass_demo = [[0, 10, 15, 25, 30, 80], [0, 0, 1, 1, 0, 0]]
notch_demo = [[0, 45, 50, 55, 80], [1, 1, 0, 1, 1]]

fig = plt.figure(figsize = (8,6))
i = sin_fftfreq > 0
sub1 = fig.add_subplot(2,2,1)
sub1.plot(sg_fftfreq[i], sg_abs_fft[i], color = 'black')
sub1.plot(lowpass_demo[0], lowpass_demo[1], color = 'red')
sub1.set_title('Lowpass filter')
sub1.set_xlabel('frequency [Hz]')
sub1.set_ylabel('amplitude')
sub1.set_xlim(0,70)
sub2 = fig.add_subplot(2,2,2)
sub2.plot(sg_fftfreq[i], sg_abs_fft[i], color = 'black')
sub2.plot(highpass_demo[0], highpass_demo[1], color = 'blue')
sub2.set_title('Highpass filter')
sub2.set_xlabel('frequency [Hz]')
sub2.set_xlim(0,70)
sub3 = fig.add_subplot(2,2,3)
sub3.plot(sg_fftfreq[i], sg_abs_fft[i], color = 'black')
sub3.plot(bandpass_demo[0], bandpass_demo[1], color = 'violet')
sub3.set_title('Bandpass filter')
sub3.set_xlabel('frequency [Hz]')
sub3.set_ylabel('amplitude')
sub3.set_xlim(0,70)
sub4 = fig.add_subplot(2,2,4)
sub4.plot(sg_fftfreq[i], sg_abs_fft[i], color = 'black')
sub4.plot(notch_demo[0], notch_demo[1], color = 'green')
sub4.set_title('Notch filter')
sub4.set_xlabel('frequency [Hz]')
sub4.set_xlim(0,70)
plt.subplots_adjust(hspace=0.4)
plt.show()

plot(fig)

```
@Pyodide.eval


{{1}}
****************************************************************

**Some annotations**

* The only one we can't use here is the highpass, because it would not mute the noise and keep the signal.
+ In this simple example most of the other 3 filters types will give the same nearly perfect result.
- Here we will only deal with a lowpass filter, because it is easiest to implement it ourselves. (LiaScript unfortunately does not understand *SciPy.signal* (yet), so we can't use ready made filters.)

****************************************************************

### Filtering in SciPy.signal

**Lowpass filtering**

```python
N = 15
Wn = 40
low = sg.butter(N, Wn, 'lowpass', fs=sampleRate, output='sos') # use a butterworth filter as lowpass
filtered_low = sg.sosfilt(low, sgSum)

```


{{1}}
****************************************************************

**Bandpass filtering**

```python
N = 20
Wn = [5,45]
band = sg.butter(N, Wn , 'bandpass', fs=sampleRate, output='sos') # use a butterworth filter as bandpass
filtered_band = sg.sosfilt(band, sgSum)

```

****************************************************************

{{2}}
****************************************************************

**Notch filtering**

```python
frq_to_filt = 50
Q = 2
b, a = sg.iirnotch(frq_to_filt, Q , fs=sampleRate) # a notch filter
filtered_notch = sg.lfilter(b, a, sgSum)

```

****************************************************************

### An easy (and very unrealistic) implementation of lowpass filters

**Create filter coefficients**

For real applications use, for example, *pyFDA* (https://github.com/chipmuenk/pyfda) to create filter coefficients.
Here it is done by designing a rectangular window and do FFT over it (as done in modul "Zeitreihenanalyse", TU Bergakademie Freiberg, Freiberg, summer semester 2019).

```python
def genCoeffsLowP( numTaps, fc, fs ):
    '''
    Creating a field of coefficients for FIR lowpass filter.
    No usage of window function, frequency response is chopped down.

    Parameters
    ----------
    numTaps : int
        number of coefficients to be created
    fc : double
        the filter's base frequency
    fs : double
        used sample rate

    Returns
    -------
    rv : ndarray
        field with filter coefficients

    '''

    fr = fc/fs

    c = np.zeros( numTaps )
    end = int(numTaps * fr)

    for i in range(0,end+1):
        c[i] = 1

    c_fft = np.fft.fft(c).real

    rv_1 = c_fft[len(c_fft)//2:]
    rv_2 = c_fft[0:len(c_fft)//2]
    rv = np.hstack((rv_1, rv_2))

    # normalization
    factor = sum(rv)
    factor /= 2
    rv /= factor

    return rv

```
@Pyodide.eval

{{1}}
****************************************************************

**Apply filter**
**... and do some other computations to compare before and after**

```python
c2 = genCoeffsLowP( 40, fc=35, fs=sampleRate )

sgFiltered = np.convolve( sgSum, c2  )

temp_fft = fp.fft(sgFiltered)
filtAbs = np.abs(temp_fft)
filtAbs /= (len(filtAbs)/2)
filtFrq = fp.fftfreq(len(filtAbs), 1/sampleRate)
i = filtFrq > 0

```
@Pyodide.eval


****************************************************************

### Results of filtering

--{{0}}--
We eliminated the disturbing frequency and kept our original signal. You may play around with the filter's settings to see some differences.

```python
fig2, ax = plt.subplots(4, 1, figsize=(8, 10))

ax[0].plot(c2, label='Coefficients', lw=0.5)
ax[0].set_title("Coefficients")
ax[0].grid()

ax[1].plot(sgSum , label='Input signal', lw=0.5)
ax[1].set_title("Input signal")

ax[2].plot(sgFiltered , label='filtered', lw=0.5)
ax[2].set_title("Lowpass applied")

ax[3].plot(filtFrq[i],filtAbs[i] , label='filtered', lw=0.8)
ax[3].set_title("FFT of filtered signal")
ax[3].set_xlim(0,100)

plt.subplots_adjust(hspace=0.4)
plt.show()

plot(fig2)

```
@Pyodide.eval


## Cross correlation

$\Rightarrow $ compare two time series and compute where they are most similar

Following situation to imagine: You have field measurements from Vibroseis Truck input recorded by geophones. The sweep goes into the underground and is reflected by every underground layer. With depth the signal looses strength and all reflections overlap each other. So when does every reflection start?

### Simulate our given data

Just execute the code and have a look at the resulting data!

**Original sweep emitted by (imaginary) Vibroseis Truck**

```python
endTime = 3 #seconds
t = np.linspace(0, endTime, endTime*sampleRate, endpoint = True)
x = np.arange(len(t))

# original sweep
sgSweep = sg.chirp(t, f0=10, f1=200, t1=endTime, method='linear')
envelope = 1 * np.sin(2 * np.pi * 0.5 * (1/endTime) * x/sampleRate)
sgSweep *= envelope

```
@Pyodide.eval

{{1}}
****************************************************************

**Single reflections to add**

```python
sgSweep1 = sgSweep * 0.9
sgSweep2 = sgSweep * 0.6
sgSweep3 = sgSweep * 0.4

```
@Pyodide.eval

****************************************************************

{{2}}
****************************************************************

**Concatenate all reflections to one resulting signal**

```python
t1 = 0.1
tarr1 = np.linspace(0, t1, int(t1*sampleRate), endpoint = True)
sg1part1 = tarr1 * 0
sg1 = np.concatenate([sg1part1,sgSweep1])

t2 = 0.74
tarr2 = np.linspace(0, t2, int(t2*sampleRate), endpoint = True)
sg2part1 = tarr2 * 0
sg2 = np.concatenate([sg2part1,sgSweep2])

t3 = 1.54
tarr3 = np.linspace(0, t3, int(t3*sampleRate), endpoint = True)
sg3part1 = tarr3 * 0
sg3 = np.concatenate([sg3part1,sgSweep3])

maximum = max(len(sg1),len(sg2),len(sg3))
arr1 = np.zeros(maximum-len(sg1))
arr2 = np.zeros(maximum-len(sg2))
arr3 = np.zeros(maximum-len(sg3))

sg1 = np.concatenate([sg1,arr1])
sg2 = np.concatenate([sg2,arr2])
sg3 = np.concatenate([sg3,arr3])

sgSum = sg1 + sg2 + sg3

```
@Pyodide.eval

****************************************************************

{{3}}
****************************************************************

**The data simulated**

```python
fig = plt.figure(figsize=(14,4))
dt = 1/sampleRate
xarr = np.linspace(0,int(len(sgSum))*dt, int(len(sgSum)))
plt.plot(xarr, sgSum)
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show()

plot(fig)

```
@Pyodide.eval



****************************************************************

### The correlation

$\Rightarrow $ compare the data with the original emitted sweep which you know


You see, the peaks are exactly located where *t1*, *t2* and *t3* (the starting times of each single reflection) were defined.

```python
corr = np.correlate(sgSum, sgSweep, "valid")

fig = plt.figure()
tarr = np.linspace(0,int(len(corr))*dt, int(len(corr)))
plt.plot(tarr,corr)
plt.xlabel("time [s]")
plt.ylabel("some amplitude")
plt.show()

plot(fig)

```
@Pyodide.eval

## Some annotations

This was just a very brief introduction into some basic processes of digital signal processing. There is much more that you may do using Python modules.

You may try out some other filters and parameters (Butterworth, Notch etc.). They may also be used, for example, in acoustics.

And I can recommend to read and understand how FFT works, because that is important in many fields of application. FFT can, for example, also be used for digital image processing to "cut out" some noise.
