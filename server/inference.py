import torch
import librosa
from pathlib import Path, PosixPath

genre_merge_dict = {
    'Rock': ['rock','alternative', 'indie', 'alternative rock', 'classic rock', 'indie rock', 'Progressive rock'],
    'Pop': ['pop', 'indie pop'],
    'Jazz / Soul': ['jazz', 'soul'],
    'Electro / Dance': ['electronic', 'dance', 'House'],
    'Disco / Funk': ['funk', 'soul', 'dance', 'soul'],
    'Hip Hop / Rnb': ['Hip-Hop', 'rnb'], 
    'Ambient / Chillout' : ['chillout', 'easy listening', 'ambient', 'experimental'], 
    'Blues': ['blues'], 
    'Folk / Country': ['folk', 'country'], 
    'Hard Rock / Metal': ['metal', 'hard rock', 'heavy metal', 'punk']
}


class Predictor(object):
    def __init__(self, ts_model_path:str, output_classes_file:str, input_length:str=59049) -> None:
        self.model_path = ts_model_path
        self.classes = []
        self.input_length = input_length
        # Load model
        try :
            self.model = torch.jit.load(ts_model_path)
            print(f"Loaded TorchScript model from {self.model_path}")
        except Exception as e:
            print(e)
        # Load model classes
        try:
            with open(output_classes_file, 'r') as f:
                for i in f.readlines():
                    self.classes.append(i.replace("\n", ""))
        except Exception as e:
            print(e)
        
    def predict(self, from_file:PosixPath=Path('mnt/Desktop/input_buffer.wav')):
        # Load buffer
        input_buffer, _ = librosa.load(from_file, sr=16000, mono=True)
        
        # Create model input
        try:
            x = torch.from_numpy(input_buffer[0:self.input_length]).unsqueeze(0)
        except Exception as e:
            print(e)
        
        # Inference
        with torch.no_grad():
            out = self.model(x)
            
        # Store results in dict 
        result_dict = {}
        for i, cl in enumerate(self.classes):
            result_dict[cl] = round(float(out[0][i]), 3)
            
        # Merge genres results
        merged_results = {}
        for k, v in genre_merge_dict.items():
            merged_results[k] = 0
            for g in v:
                merged_results[k] += round(result_dict[g] * 1/len(v), 3)
        
        return merged_results
