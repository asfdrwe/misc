import gradio
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), 'ComfyUI'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ComfyUI/comfy'))

import main

from ComfyUI.nodes import CheckpointLoaderSimple, CLIPTextEncode, VAEDecode, KSampler, SaveImage, VAELoader, LoraLoader, VAEEncodeForInpaint
from ComfyUI.comfy_extras.nodes_mask import ImageToMask

import torch
import numpy as np
import random
from PIL import Image
import torch

def generate(image, positive, denoise):
    model, clip, vae = CheckpointLoaderSimple().load_checkpoint("waiNSFWIllustrious_v140.safetensors")
    loramodel, loraclip = LoraLoader().load_lora(model, clip, "dmd2_sdxl_4step_lora_fp16.safetensors", 1.0, 1.0)

    pixels = np.array(image['background']).astype(np.float32) / 255.0
    pixels = torch.from_numpy(pixels)[None,]
    m = np.array(image['layers'][0].convert("RGBA")).astype(np.float32) / 255.0
    m = torch.from_numpy(m)[None,]
    mask = ImageToMask().image_to_mask(m, "red")

    latent = VAEEncodeForInpaint().encode(vae, pixels, mask)[0]

    positive = CLIPTextEncode().encode(loraclip, positive)[0]
    negative = CLIPTextEncode().encode(loraclip, "")[0]

    with torch.no_grad():
        denoised = KSampler().sample(loramodel, random.randint(0, sys.maxsize), 8, 1.0, "lcm", "simple", positive, negative, latent, denoise)[0]
        images = VAEDecode().decode(vae, denoised)[0]
    SaveImage().save_images(images)
    return Image.fromarray((images[0] * 255).numpy().astype(np.uint8))

def main():
    image    = gradio.ImageEditor(type='pil', brush = gradio.Brush(colors = "black"),  label = "入力画像")
    positive = gradio.TextArea(label = "ポジティブプロンプト")
    denoise  = gradio.Number(label = "デノイズ", value = 0.70)

    app = gradio.Interface(
      fn = generate, 
      title = "画像生成", 
      inputs = [image, positive, denoise], 
      outputs = "image", submit_btn = "生成", clear_btn = None, flagging_mode = "never")

    app.launch(inbrowser = True)

if __name__ == "__main__":
    main()
