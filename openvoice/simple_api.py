import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter


class SimpleVoiceApi:
    def __init__(self, config_path, model_path, device="cpu"):
        self.device = device
        self.model_path = model_path
        self.converter = ToneColorConverter(config_path, device)
        
        self.converter.load_ckpt(model_path)
        
        
    def convert_voice(self, src_audio, target_voice, save_path):
        
        print("Extracting source embedding...")
        source_se = self.converter.extract_se(src_audio)
        
        print("Extracting target embedding...")
        target_se = self.converter.extract_se(target_voice)
        
        
        print("Converting voice...")
        encode_message = "@MyShell"
        
        self.converter.convert(
            audio_src_path=src_audio,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message
            )
        print("Done 👍!!!")
