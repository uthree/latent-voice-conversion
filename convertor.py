import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoderResBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.res_conv nn.Conv1d(input_channels, output_channels, 1, 1, 0, bias=False)
        self.pool = nn.AvgPool1d(2)
        self.conv1 = nn.Conv1d(input_channels, input_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(input_channels, output_channels, 1, 1, 0)

    def forward(self, x):
        res = self.pool(self.res_conv(x))
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x + res


class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_mels=80,
                n_fft=1024,
                dim_spk)
        self.layers = nn.Sequential(
                SpeakerEncoderResBlock(80, 32),
                SpeakerEncoderResBlock(32, 64),
                SpeakerEncoderResBlock(64, 128),
                SpeakerEncoderResBlock(128, 256),
                SpeakerEncoderResBlock(256, 512))
        self.output_layer = nn.Conv1d(512, dim_spk, 1, 1, 0)

    def forward(self, x):
        x = self.to_mel(x)
        x = self.layers(x)
        x = self.output_layer(x)
        return x
