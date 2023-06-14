import torch
import torchaudio
import glob
from tqdm import tqdm
import os


class WaveFileDirectoryWithClass(torch.utils.data.Dataset):
    def __init__(self, source_dir_paths=[], length=32768, max_files=-1, sampling_rate=16000):
        super().__init__()
        print("Loading Data")
        self.path_list = []
        self.data = []
        self.speaker_ids = []
        self.speakers = []
        formats = ["mp3", "wav", "ogg"]
        print("Getting paths")
        for dir_path in source_dir_paths:
            for fmt in formats:
                self.path_list += glob.glob(os.path.join(dir_path, f"**/*.{fmt}"), recursive=True)
        if max_files != -1:
            self.path_list = self.path_list[:max_files]
        print("Chunking")
        for path in tqdm(self.path_list):
            tqdm.write(path)
            wf, sr = torchaudio.load(path) # wf.max() = 1 wf.min() = -1
            # Resample
            wf = torchaudio.functional.resample(wf, sr, sampling_rate)
            # Chunk
            waves = torch.split(wf, length, dim=1)
            tqdm.write(f"    Loading {len(waves)} data...")
            for w in waves:
                if w.shape[1] == length:
                    self.data.append(w[0])
                    dir_name = os.path.dirname(path)
                    if dir_name not in self.speakers:
                        self.speakers.append(dir_name)
                    speaker_id = self.speakers.index(dir_name)
                    self.speaker_ids.append(speaker_id)
        self.length = length
        print(f"Loaded total {len(self.data)} data.")

    def __getitem__(self, index):
        return self.data[index], self.speaker_ids[index]

    def __len__(self):
        return len(self.data)
