import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from tqdm import tqdm

from dataset import WaveFileDirectoryWithClass
from vae import VAE
from convertor import VoiceConvertor


def load_or_init_models(device=torch.device('cpu')):
    vc = VoiceConvertor().to(device)
    if os.path.exists("convertor.pt"):
        vc.load_state_dict(torch.load("convertor.pt", map_location=device))
    return vc


def save_models(vc):
    torch.save(vc.state_dict(), "convertor.pt")
    print("saved models")


parser = argparse.ArgumentParser(description="Train voice conversion.")

parser.add_argument('dataset_path')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=10000, type=int)
parser.add_argument('-b', '--batch', default=64, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-m', '--maxdata', default=-1, type=int, help="max dataset size")
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)

args = parser.parse_args()
device = torch.device(args.device)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

weight_con = 0.1
weight_cyc = 0.1

ds = WaveFileDirectoryWithClass(
        [args.dataset_path],
        length=32768,
        max_files=args.maxdata)

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

vc = load_or_init_models(device)

optimizer = optim.Adam(vc.parameters(), lr=args.learning_rate)

vae = VAE().to(device)
vae.load_state_dict(torch.load('./vae.pt', map_location=device))
vae.eval()


for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))

    for batch, (wave, _) in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)
        amp = torch.rand(N, 1).to(device) * 0.75 + 0.25
        wave = wave * amp
        x = vae.encode(wave)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            spk = vc.speaker_encoder(wave)
            spk_tgt = torch.roll(spk, dims=0, shifts=1)
            c = vc.content_encoder(x)
            rec_out = vc.decoder(c, spk)
            loss_rec = (rec_out - x).abs().mean()
            converted = vc.decoder(c, spk_tgt)
            cyc_out = vc.decoder(vc.content_encoder(converted), spk)
            loss_cyc = (cyc_out - x).abs().mean()
            loss_con = (vc.content_encoder(converted) - c).abs().mean()
            loss_vc = loss_rec + weight_con * loss_con

        if torch.any(torch.isnan(loss_vc)):
            exit()
            
        scaler.scale(loss_vc).backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0, 2.0)
        scaler.step(optimizer)

        if batch % 100 == 0:
            save_models(vc)

        scaler.update()
        tqdm.write(f"Rec.: {loss_rec.item():.4f}, Con.: {loss_con.item():.4f}, Cyc.: {loss_cyc.item():.4f}")
        bar.set_description(f"Loss: {loss_vc.item():.4f}")
        bar.update(N)
