import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from tqdm import tqdm

from dataset import WaveFileDirectoryWithClass
from vae import VAE, Discriminator, MelSpectrogramLoss


def load_or_init_models(device=torch.device('cpu')):
    vae = VAE().to(device)
    D = Discriminator().to(device)
    if os.path.exists("vae.pt"):
        vae.load_state_dict(torch.load("vae.pt", map_location=device))
    if os.path.exists("vae_discriminator.pt"):
        D.load_state_dict(torch.load("vae_discriminator.pt", map_location=device))
    return vae, D


def save_models(vae, D):
    torch.save(vae.state_dict(), "vae.pt")
    torch.save(D.state_dict(), "vae_discriminator.pt")
    print("saved models")


parser = argparse.ArgumentParser(description="Train VAE.")

parser.add_argument('dataset_path')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=10000, type=int)
parser.add_argument('-b', '--batch', default=2, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-m', '--maxdata', default=-1, type=int, help="max dataset size")
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('--freeze-encoder', default=False, type=bool)

args = parser.parse_args()
device = torch.device(args.device)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

weight_kl = 0.1
weight_feat = 5.0
weight_mel = 45.0


ds = WaveFileDirectoryWithClass(
        [args.dataset_path],
        length=32768,
        max_files=args.maxdata)

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

mel_loss = MelSpectrogramLoss().to(device)

vae, D = load_or_init_models(device)

OptVAE = optim.Adam(vae.parameters(), lr=args.learning_rate, betas=(0.9, 0.5))
OptD = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.9, 0.5))

if args.freeze_encoder:
    for param in vae.encoder.parameters():
        param.requires_grad = False


for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))

    for batch, (wave, _) in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)
        amp = torch.rand(N, 1).to(device) * 0.75 + 0.25
        wave = wave * amp
        
        OptVAE.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            mean, logvar = vae.encoder(wave)
            z = mean + torch.randn_like(logvar) * torch.exp(logvar)
            fake = vae.decoder(z)
            
            loss_feat = D.feat_loss(fake, wave)
            loss_mel = mel_loss(fake, wave)
            loss_kl = (-1 - logvar + torch.exp(logvar) + mean ** 2).mean()

            logits = D.logits(fake)
            loss_adv = 0
            for logit in logits:
                loss_adv += (logit ** 2).mean()
            loss_VAE = loss_adv + weight_feat * loss_feat + weight_kl * loss_kl + weight_mel * loss_mel
            
        scaler.scale(loss_VAE).backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0, 2.0)
        scaler.step(OptVAE)

        OptD.zero_grad()
        fake = fake.detach()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_D = 0
            logits = D.logits(fake)
            for logit in logits:
                loss_D += ((logit - 1) ** 2).mean()
            logits = D.logits(wave)
            for logit in logits:
                loss_D += (logit ** 2).mean()

        scaler.scale(loss_D).backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0, 2.0)
        scaler.step(OptD)

        if batch % 100 == 0:
            save_models(vae, D)

        scaler.update()
        tqdm.write(f"Adv: {loss_adv.item():.4f}, feat.: {loss_feat.item():.4f}, K.L.: {loss_kl.item():.4f}, Mel.: {loss_mel.item():.4f}")
        bar.set_description(f"G: {loss_VAE.item():.4f}, D: {loss_D.item():.4f}")
        bar.update(N)
