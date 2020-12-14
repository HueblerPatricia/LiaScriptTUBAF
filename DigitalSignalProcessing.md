<!--

author:   Patricia Huebler

email:    Patricia.Huebler@student.tu-freiberg.de

language: en

narrator: UK English Female

import:  https://github.com/LiaTemplates/Pyodide
-->

# Introduction to digital signal processing in Python 3

literature for this chapter:

José Unpingco: Python for Signal Processing. Featuring IPython Notebooks. Springer International Publishing, Switzerland 2014

See on LiaScript:

https://liascript.github.io/course/?https://raw.githubusercontent.com/HueblerPatricia/LiaScriptTUBAF/main/DigitalSignalProcessing.md

## Python modules

--{{0}}--
First we bind in some Python modules.

```python
import numpy as np
import scipy as sp
import scipy.signal as sg
import scipy.fftpack
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

## Some sample rates to choose from

  --{{0}}--
We will use this sample rate through the whole chapter.

```python
#index      =   0    1     2     3     4      5       6       7       8
sampleRates = (250, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)

sampleRate = sampleRates[5] #Hz

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

## FFT : Fast Fourier Transformation

"Fourier analysis converts a signal from its original domain (often time [...]) to a representation in the frequency domain [...]"
source: https://en.wikipedia.org/wiki/Fast_Fourier_transform

So let's do an FFT for all of our 3 signals!

```python
sin_fft = sp.fftpack.fft(sgSine)
sin_abs_fft = 2 * np.abs(sin_fft)/(len(sgSine))
sin_fftfreq = sp.fftpack.fftfreq(len(sin_abs_fft), 1/sampleRate)

sq_fft = sp.fftpack.fft(sgSquare)
sq_abs_fft = 2 * np.abs(sq_fft)/(len(sgSquare))
sq_fftfreq = sp.fftpack.fftfreq(len(sq_abs_fft), 1/sampleRate)

sweep_fft = sp.fftpack.fft(sgSweep)
sweep_abs_fft = 2 * np.abs(sweep_fft)/(len(sgSweep))
sweep_fftfreq = sp.fftpack.fftfreq(len(sweep_abs_fft), 1/sampleRate)

```
@Pyodide.eval


{{1}}
******************************************************************

**Plots**

```python
fig, ax = plt.subplots(3,1, figsize = (12,12))
i = sin_fftfreq > 0

ax[0].plot(sin_fftfreq[i], sin_abs_fft[i], lw = .7)
ax[0].set_title("Spectrum of sine signal")
ax[0].set_xlabel('frequency [Hz]')
ax[0].set_ylabel('amplitude')
ax[0].set_xlim(0,500)
ax[0].grid()

ax[1].plot(sq_fftfreq[i], sq_abs_fft[i], lw = .7)
ax[1].set_title("Spectrum of square-wave signal")
ax[1].set_xlabel('frequency [Hz]')
ax[1].set_ylabel('amplitude')
ax[1].set_xlim(0,500)
ax[1].grid()

ax[2].plot(sweep_fftfreq[i], sweep_abs_fft[i], lw = .7)
ax[2].set_title("Spectrum of sweep")
ax[2].set_xlabel('frequency [Hz]')
ax[2].set_ylabel('amplitude')
ax[2].set_xlim(0,500)
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
sampleRate = sampleRates[5]
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
sin_fft = sp.fftpack.fft(sgSine)
sin_abs_fft = 2 * np.abs(sin_fft)/(len(sgSine))
sin_fftfreq = sp.fftpack.fftfreq(len(sin_abs_fft), 1/sampleRate)

sg_fft = sp.fftpack.fft(sgSum)
sg_abs_fft = 2 * np.abs(sg_fft)/(len(sgSum))
sg_fftfreq = sp.fftpack.fftfreq(len(sg_abs_fft), 1/sampleRate)

```
@Pyodide.eval

--{{0}}--
In frequency domain we clearly see two parted frequencies. One is the signal, the other is the noise.

```python
fig = plt.figure(figsize = (12,8))
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

fig = plt.figure(figsize = (12,8))
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
- In this simple example most of the other 3 filters types will give the same nearly perfect result.

****************************************************************

### Filtering

**Lowpass filtering**

```python
N = 15
Wn = 40
low = sg.butter(N, Wn, 'lowpass', fs=sampleRate, output='sos') # use a butterworth filter as lowpass
filtered_low = sg.sosfilt(low, sgSum)

```
@Pyodide.eval


{{1}}
****************************************************************

**Bandpass filtering**

```python
N = 20
Wn = [5,45]
band = sg.butter(N, Wn , 'bandpass', fs=sampleRate, output='sos') # use a butterworth filter as bandpass
filtered_band = sg.sosfilt(band, sgSum)

```
@Pyodide.eval

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
@Pyodide.eval

****************************************************************

### Results of filtering

--{{0}}--
With all three filters we eliminated the disturbing frequency and kept our original signal. You may play around with the filter's settings to see some diferences.

```python
fig = plt.figure(figsize = (12,8))
sub1 = fig.add_subplot(2,2,1)
sub1.plot(t, sgSum)
sub1.set_title('Before')
sub1.set_xlabel('time [s]')
sub1.set_ylabel('amplitude')
sub2 = fig.add_subplot(2,2,2)
sub2.plot(t, filtered_low)
sub2.set_title('Lowpass filter applied')
sub2.set_xlabel('time [s]')
sub3 = fig.add_subplot(2,2,3)
sub3.plot(t, filtered_band)
sub3.set_title('Bandpass filter applied')
sub3.set_xlabel('time [s]')
sub4 = fig.add_subplot(2,2,4)
sub4.plot(t, filtered_notch)
sub4.set_title('Notch filter applied')
sub4.set_xlabel('time [s]')
plt.subplots_adjust(hspace=0.4)
plt.show()

plot(fig)

```
@Pyodide.eval


## Cross correlation

$\Rightarrow $ compare two time series and compute where they are most similar

Following situation to imagine: You have field measurements from Vibroseis Truck input recorded by geophones. The sweep goes into the underground and is reflected by every underground layer. With depth the signal looses strength and all reflections overlap each other. So when does every reflection start?

### Simulate our given data

Just execute the code and have a look at the resulting data!

**Original sweep emitted by Vibroseis Truck**

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

--{{0}}--
The correlation maxima are exactly located at the points in time, where a new reflected sweep starts.

## Some annotations

This was just a very brief introduction into some basic processes of digital signal processing. There is much more that you may do using Python modules.
You may try out some other filters and parameters. They may also be used, for example, in acoustics.

And I can recommend to read and understand how FFT works, because that is important in many fields of application. FFT can, for example, also be used for digital image processing to "cut out" some noise.
