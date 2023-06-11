import torch
import clip
from PIL import Image

def get_clip_model_and_processor(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.to(torch.float32)
    return model,preprocess
