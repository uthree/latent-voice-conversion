import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SpeakerEncoderResBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.res_conv = nn.Conv1d(input_channels, output_channels, 1, 1, 0, bias=False)
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
    def __init__(self, dim_spk=128):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_mels=80,
                n_fft=1024)
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
        x = x.mean(dim=2, keepdim=True)
        return x


class ContentEncoderResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.conv2 = nn.Conv1d(channels, channels, 5, 1, 2)

    def forward(self, x):
        return self.conv2(F.gelu(self.conv1(F.gelu(x)))) + x



class ContentEncoder(nn.Module):
    def __init__(self, input_channels=192, internal_channels=256, bottleneck=4, num_layers=8):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, internal_channels, 1, 1, 0)
        self.mid_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.mid_layers.append(ContentEncoderResBlock(internal_channels))
        self.output_layer = nn.Conv1d(internal_channels, bottleneck, 1, 1, 0)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class DecoderResBlock(nn.Module):
    def __init__(self, channels, dim_spk=128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.conv2 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.to_mu = nn.Conv1d(dim_spk, channels, 1, 1, 0)
        self.to_sigma = nn.Conv1d(dim_spk, channels, 1, 1, 0)

    def forward(self, x, spk):
        res = x
        x = self.conv2(F.gelu(self.conv1(F.gelu(x))))
        x = x * self.to_sigma(spk) + self.to_mu(spk)
        return x + res


class Decoder(nn.Module):
    def __init__(self, bottleneck=4, output_channels=192, internal_channels=256, num_layers=8, dim_spk=128):
        super().__init__()
        self.input_layer = nn.Conv1d(bottleneck, internal_channels, 1, 1, 0)
        self.mid_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.mid_layers.append(DecoderResBlock(internal_channels, dim_spk))
        self.output_layer = nn.Conv1d(internal_channels, output_channels, 1, 1, 0)

    def forward(self, x, spk):
        x = self.input_layer(x)
        for layer in self.mid_layers:
            x = layer(x, spk)
        x = self.output_layer(x)
        return x


class VoiceConvertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = ContentEncoder()
        self.decoder = Decoder()
        self.speaker_encoder = SpeakerEncoder()
