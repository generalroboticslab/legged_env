import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import torch
# from .common.buffer import BatchedRingBufferWithSharedCounter, RingBufferCounter
from common.buffer import RingTensorBuffer

# y[n] = b0*x[n] + b1*x[n-1] + ... + bk*x[n-k] - a1*y[n-1] - ... - ak*y[n-k]


cut_off_frequency = 10
fs = 50
filter_order = 2 # Filter order

b, a = signal.butter(N=filter_order, Wn=cut_off_frequency, btype='low', analog=False,fs=50)

# # Frequency Axis
# w, h = signal.freqz(b, a)

# # Plotting
# plt.figure()

# # Magnitude Response
# plt.subplot(2, 1, 1)
# plt.plot(w/np.pi, 20 * np.log10(abs(h)), 'b')
# plt.axvline(cut_off_frequency/(fs/2), color='green') # cutoff frequency
# plt.title('Butterworth Lowpass Filter Frequency Response')
# plt.ylabel('Amplitude [dB]')
# plt.grid()


# # Phase Response
# plt.subplot(2, 1, 2)
# plt.plot(w/np.pi, np.unwrap(np.angle(h))*180/np.pi, 'g')
# plt.axvline(cut_off_frequency/(fs/2), color='green') # cutoff frequency
# plt.ylabel('Phase [degrees]')
# plt.xlabel('Frequency [rad/sample]')
# plt.grid()

# plt.show()


a = torch.tensor(a)
b = torch.tensor(b)


x = RingTensorBuffer(buffer_len=filter_order+1,shape=1) # Input sample buffer
y = RingTensorBuffer(buffer_len=filter_order+1,shape=1)# Output sample buffer

# x = [0] * filter_order    # Input sample buffer
# y = [0] * filter_order    # Output sample buffer

# Generate Test Signal (Simulating a Continuous Signal with Noise)
# t = np.linspace(0, 1, fs+1, False)
# sig = np.sin(2*np.pi*2*t) + 0.1*np.sin(2*np.pi*20*t) + 0.05 * np.random.randn(fs+1)  # Added some noise

t = torch.linspace(0, 1, fs+1)
sig  = torch.sin(2*np.pi*2*t) + 0.1*torch.sin(2*np.pi*20*t) + 0.05 * torch.randn(fs+1) # Added some noise

for k in range(filter_order):
    x.add(0)
    y.add(0)

# Filtering in Real-Time
filtered_signal = []
for sample in sig:

    # x.pop(0)
    # x.append(sample)
    # filtered_sample = sum(b[i] * x[-i] for i in range(filter_order+1))
    # filtered_sample -= sum(a[i] * y[-i] for i in range(1, filter_order+1))
    # y.pop(0)
    # y.append(filtered_sample)

    x.add(sample)

    filtered_sample = (b*x[:].ravel()).sum() - (a[1:filter_order+1]*y[0:filter_order].ravel()).sum()

    y.add(filtered_sample)

    filtered_signal.append(filtered_sample)

# Plotting
plt.figure(figsize=(10, 6))  # Adjust the figure size for better visualization
plt.plot(t, sig, label='Original Signal', alpha=0.7)  # Make the original signal slightly transparent
plt.plot(t, filtered_signal, label=f'Filtered Signal ({cut_off_frequency} Hz Low-pass)', linewidth=2)
plt.title('Signal Filtering with Butterworth Low-Pass Filter (Zero-Phase)')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(alpha=0.4)  # Add a subtle grid
plt.tight_layout()
plt.show()