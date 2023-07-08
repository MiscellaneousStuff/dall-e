from vqgan_jax.modeling_flax_vqgan import VQModel

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import requests, io

def get_image(path):
    return Image.open(path)

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def preprocess(img, target_image_size=256,):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img.permute(0, 2, 3, 1).numpy()

def custom_to_pil(x):
    x = np.clip(x, -1., 1.)
    x = (x + 1.)/2.
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def display(rec):
    before_process = rec[0]
    after_process  = preprocess_vqgan(before_process)
    return custom_to_pil(np.array(after_process))

def recon(url, vqgan_model, size=384):
    size=size
    if url.startswith("http"):
        image = download_image(url)
    else:
        image = get_image(url)
    image = preprocess(image, size)
    quant_states, indices = vqgan_model.encode(image)
    print(quant_states.shape, indices.shape)
    # rec                   = vqgan_model.decode(quant_states)
    rec                   = vqgan_model.decode_code(indices)
    return quant_states, indices, rec

class VQGAN(object):
    def __init__(self, repo, commit_id):
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            repo, revision=commit_id, _do_init=False)
    
    def encode(self, image):
        quant_states, indices = self.vqgan.encode(image, params=self.vqgan_params)
        return quant_states, indices

    def decode(self, quant_states):
        rec = self.vqgan.decode(quant_states, params=self.vqgan_params)
        return rec
    
    def decode_code(self, indices):
        rec = self.vqgan.decode_code(indices, params=self.vqgan_params)
        return rec