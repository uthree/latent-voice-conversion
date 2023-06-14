import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
import torchaudio


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int(((kernel_size -1)*dilation)/2)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([])
        self.convs2 = nn.ModuleList([])

        for d in dilation:
            self.convs1.append(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                        padding=get_padding(kernel_size, d)))
            self.convs2.append(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                        padding=get_padding(kernel_size, d)))

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(x)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(x)
            x = xt + x
        return x


class MRF(nn.Module):
    def __init__(self,
            channels,
            kernel_sizes=[3, 7, 11],
            dilation_rates=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for k, d in zip(kernel_sizes, dilation_rates):
            self.blocks.append(
                    ResBlock(channels, k, d))

    def forward(self, x):
        for block in self.blocks:
            xt = block(x)
            x = xt + x
        return x


class EncoderResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.c2 = nn.Conv1d(channels, channels, 5, 1, 2)

    def forward(self, x):
        res = x
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c1(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c2(x)
        return x + res


class Encoder(nn.Module):
    def __init__(self, output_channels=192, num_layers=6):
        super().__init__()
        self.to_spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, normalized=True)
        self.first_layer = nn.Conv1d(513, output_channels, 5, 1, 2)
        self.mid_layers = nn.Sequential(*[EncoderResBlock(output_channels) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(output_channels, output_channels * 2, 1, 1, 0)
        self.apply(init_weights)

    def forward(self, x):
        x = self.to_spectrogram(x)[:, :, 1:]
        x = self.first_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self,
            input_channels=192,
            upsample_initial_channels=512,
            deconv_strides=[8, 8, 2, 2],
            deconv_kernel_sizes=[16, 16, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_rates=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            ):
        super().__init__()
        self.pre = nn.Conv1d(input_channels, upsample_initial_channels, 7, 1, 3)

        self.ups = nn.ModuleList([])
        for i, (s, k) in enumerate(zip(deconv_strides, deconv_kernel_sizes)):
            self.ups.append(
                    nn.ConvTranspose1d(
                        upsample_initial_channels // (2 ** i),
                        upsample_initial_channels // (2 ** ( i + 1 )),
                        k, s, (k-s)//2))

        self.MRFs = nn.ModuleList([])
        for i in range(len(self.ups)):
            c = upsample_initial_channels//(2**(i+1))
            self.MRFs.append(MRF(c, resblock_kernel_sizes, resblock_dilation_rates))

        self.post = nn.Conv1d(c, 1, 7, 1, 3, bias=False)
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.pre(x)
        for up, MRF in zip(self.ups, self.MRFs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = MRF(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.post(x)
        x = torch.tanh(x)
        x = x.squeeze(1)
        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        mu, sigma = self.encoder(x)
        return mu

    def decode(self, z):
        y = self.decoder(z)
        return y


class PeriodicDiscriminator(nn.Module):
    def __init__(self,
                 channels=32,
                 period=2,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 dropout_rate=0.2,
                 groups = []
                 ):
        super().__init__()
        self.input_layer = nn.utils.spectral_norm(
                nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), 0))
        self.layers = nn.Sequential()
        for i in range(num_stages):
            c = channels #* (2 ** i)
            c_next = channels #* (2 ** (i+1))
            if i == (num_stages - 1):
                self.layers.append(
                        nn.utils.spectral_norm(
                            nn.Conv2d(c, c, (kernel_size, 1), (stride, 1), groups=groups[i])))
            else:
                self.layers.append(
                        nn.utils.spectral_norm(
                            nn.Conv2d(c, c_next, (kernel_size, 1), (stride, 1), groups=groups[i])))
                self.layers.append(
                        nn.Dropout(dropout_rate))
                self.layers.append(
                        nn.LeakyReLU(LRELU_SLOPE))
        c = channels #* (2 ** (num_stages-1))
        self.final_conv = nn.utils.spectral_norm(
                nn.Conv2d(c, c, (5, 1), 1, 0)
                )
        self.final_relu = nn.LeakyReLU(LRELU_SLOPE)
        self.output_layer = nn.utils.spectral_norm(
                nn.Conv2d(c, 1, (3, 1), 1, 0))
        self.period = period

    def forward(self, x, logit=True):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.final_conv(x)
        x = self.final_relu(x)
        if logit:
            x = self.output_layer(x)
        return x

    def feat(self, x):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class MultiPeriodicDiscriminator(nn.Module):
    def __init__(self,
                 periods=[2, 3, 5, 7, 11],
                 groups=[1, 1, 1, 1],
                 channels=64,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])

        for p in periods:
            self.sub_discriminators.append(
                    PeriodicDiscriminator(channels,
                                          p,
                                          kernel_size,
                                          stride,
                                          num_stages,
                                          groups=groups))

    def forward(self, x):
        logits = []
        for sd in self.sub_discriminators:
            logits.append(sd(x))
        return logits
    
    def feat(self, x):
        feats = []
        for sd in self.sub_discriminators:
            feats = feats + sd.feat(x)
        return feats


class ScaleDiscriminator(nn.Module):
    def __init__(
            self,
            segment_size=16,
            channels=[64, 64, 64],
            norm_type='spectral',
            kernel_size=11,
            strides=[1, 1, 1],
            dropout_rate=0.1,
            groups=[],
            pool = 1
            ):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(pool)
        self.segment_size = segment_size
        if norm_type == 'weight':
            norm_f = nn.utils.weight_norm
        elif norm_type == 'spectral':
            norm_f = nn.utils.spectral_norm
        else:
            raise f"Normalizing type {norm_type} is not supported."
        self.layers = nn.Sequential()
        self.input_layer = norm_f(nn.Conv1d(segment_size, channels[0], 1, 1, 0))
        for i in range(len(channels)-1):
            if i == 0:
                k = 15
            else:
                k = kernel_size
            self.layers.append(
                    norm_f(
                        nn.Conv1d(channels[i], channels[i+1], k, strides[i], 0, groups=groups[i])))
            self.layers.append(
                    nn.Dropout(dropout_rate))
            self.layers.append(nn.LeakyReLU(LRELU_SLOPE))
        self.output_layer = norm_f(nn.Conv1d(channels[-1], 1, 1, 1, 0))

    def forward(self, x, logit=True):
        # Padding
        if x.shape[1] % self.segment_size != 0:
            pad_len = self.segment_size - (x.shape[1] % self.segment_size)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)
        x = x.view(x.shape[0], self.segment_size, -1)
        x = self.pool(x)
        x = self.input_layer(x)
        x = self.layers(x)
        if logit:
            x = self.output_layer(x)
        return x

    def feat(self, x):
        # Padding
        if x.shape[1] % self.segment_size != 0:
            pad_len = self.segment_size - (x.shape[1] % self.segment_size)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)
        x = x.view(x.shape[0], self.segment_size, -1)
        x = self.pool(x)
        x = self.input_layer(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            segments=[1, 1, 1],
            channels=[64, 64, 64, 64, 64, 64],
            kernel_sizes=[15, 41, 41, 41, 41, 41],
            strides=[1, 2, 4, 4, 4, 4],
            groups=[1, 1, 1, 1, 1, 1],
            pools=[1, 2, 4]
            ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])
        for i, (k, sg, p) in enumerate(zip(kernel_sizes, segments, pools)):
            if i == 0:
                n = 'spectral'
            else:
                n = 'weight'
            self.sub_discriminators.append(
                    ScaleDiscriminator(sg, channels, n, k, strides, groups=groups, pool=p))

    def forward(self, x):
        logits = []
        for sd in self.sub_discriminators:
            logits.append(sd(x))
        return logits

    def feat(self, x):
        feats = []
        for sd in self.sub_discriminators:
            feats = feats + sd.feat(x)
        return feats


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.MPD = MultiPeriodicDiscriminator()
        self.MSD = MultiScaleDiscriminator()

    def logits(self, x):
        return self.MPD(x) + self.MSD(x)

    def feat_loss(self, fake, real):
        with torch.no_grad():
            real_feat = self.MPD.feat(real) + self.MSD.feat(real)
        fake_feat = self.MPD.feat(fake) + self.MSD.feat(fake)
        loss = 0
        for r, f in zip(real_feat, fake_feat):
            loss = loss + F.l1_loss(f, r)
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=22050, n_ffts=[512, 1024, 2048], n_mels=80, normalized=False):
        super().__init__()
        self.to_mels = nn.ModuleList([])
        for n_fft in n_ffts:
            self.to_mels.append(torchaudio.transforms.MelSpectrogram(sample_rate,
                                                                n_mels=n_mels,
                                                                n_fft=n_fft,
                                                                normalized=normalized,
                                                                hop_length=256))

    def forward(self, fake, real):
        loss = 0
        for to_mel in self.to_mels:
            to_mel = to_mel.to(real.device)
            with torch.no_grad():
                real_mel = to_mel(real)
            loss += F.l1_loss(to_mel(fake), real_mel).mean() / len(self.to_mels)
        return loss
