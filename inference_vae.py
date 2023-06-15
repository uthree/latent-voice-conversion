import argparse
import os

import torch
import torchaudio
from torchaudio.functional import resample as resample

from vae import VAE


parser = argparse.ArgumentParser(description="Inference")

parser.add_argument('-d', '--device', default='cpu',
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-i', '--input', default='./inputs',
                    help="Input directory")
parser.add_argument('-o', '--output', default='./outputs',
                    help="Output directory")
args = parser.parse_args()

device = torch.device(args.device)

vae = VAE().to(device)
vae.load_state_dict(torch.load('./vae.pt', map_location=device))

if not os.path.exists(args.output):
    os.mkdir(args.output)


for i, fname in enumerate(os.listdir(args.input)):
    print(f"Inferencing {fname}")
    with torch.no_grad():
        wf, sr = torchaudio.load(os.path.join(args.input, fname))
        wf = resample(wf, sr, 22050)
        wf = wf.to(device)
        
        z = vae.encode(wf)
        wf = vae.decode(z)

        wf = resample(wf, 22050, sr)
        wf = wf.to(torch.device('cpu'))
        out_path = os.path.join(args.output, f"output_{fname}_{i}.wav")
        torchaudio.save(out_path, src=wf, sample_rate=sr)



