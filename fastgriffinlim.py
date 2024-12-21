import numpy as np
import torch
import torchaudio
import time
import torch.nn as nn

# Benchmarking configurations
n_fft = 1024
n_iter = 50
hop_length = 256
momentum = 0.99
device = "cuda"

# Simple input: a small random spectrogram
F, T = 513, 100
S_magnitude = torch.rand((F, T), device=device)

# Initialize Fast Griffin-Lim
class FastGriffinLim(nn.Module):
    def __init__(self, n_fft, n_iter, hop_length=None, momentum=0.99):
        super(FastGriffinLim, self).__init__()
        self.n_fft = n_fft
        self.n_iter = n_iter
        self.hop_length = hop_length or n_fft // 4
        self.momentum = momentum

    def _stft(self, x):
        return torch.stft(x, self.n_fft, hop_length=self.hop_length, return_complex=True)

    def _istft(self, X):
        return torch.istft(X, self.n_fft, hop_length=self.hop_length, length=None)

    def _projection(self, S_magnitude, Y):
        # Match the dimensions of S_magnitude and Y
        min_time_dim = min(S_magnitude.size(-1), Y.size(-1))
        S_magnitude = S_magnitude[..., :min_time_dim]
        Y = Y[..., :min_time_dim]
        return S_magnitude * torch.exp(1j * torch.angle(Y))

    def forward(self, S_magnitude, init_phase=None):
        device = S_magnitude.device
        if init_phase is not None:
            init_phase = init_phase.to(device)
        if init_phase is None:
            init_phase = torch.rand_like(S_magnitude, device=device) * 2 * np.pi
        C_prev = S_magnitude * torch.exp(1j * init_phase)
        time_signal = torch.randn((self.n_fft * 2,), device=device)  # Length >= n_fft
        T_prev = self._istft(self._stft(time_signal))
        C_prev = self._stft(T_prev)
        for n in range(self.n_iter):
            T = self._istft(C_prev)
            C = self._stft(T)
            T_proj = self._istft(self._projection(S_magnitude, C))
            C_proj = self._stft(T_proj)
            C_next = C_proj + self.momentum * (C_proj - C_prev)
            C_prev = C_next
        return self._istft(C_next)

fgl = FastGriffinLim(n_fft=n_fft, n_iter=n_iter, hop_length=hop_length, momentum=momentum).to(device)

# Initialize torchaudio Griffin-Lim
torch_griffinlim = torchaudio.transforms.GriffinLim(
    n_fft=n_fft, n_iter=n_iter, hop_length=hop_length, power=1.0, rand_init=False
).to(device)

# Benchmark Fast Griffin-Lim
torch.cuda.synchronize()
start_time = time.time()
fgl_waveform = fgl(S_magnitude)
torch.cuda.synchronize()
fgl_time = time.time() - start_time

# Benchmark torchaudio Griffin-Lim
torch.cuda.synchronize()
start_time = time.time()
torch_griffinlim_waveform = torch_griffinlim(S_magnitude)
torch.cuda.synchronize()
torch_griffinlim_time = time.time() - start_time

# Output results
print(fgl_time, torch_griffinlim_time)
