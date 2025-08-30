import gradio
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), 'ComfyUI'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ComfyUI/comfy'))

import main, folder_paths

from ComfyUI.nodes import CheckpointLoaderSimple, CLIPTextEncode, EmptyLatentImage, VAEDecode, KSampler, SaveImage, VAELoader, LoraLoader

import torch
import numpy as np
import random
from PIL import Image

def generate(positive, negative, width, height, step, cfg, sampler, scheduler, lora, lorastr):
    model, clip, vae = CheckpointLoaderSimple().load_checkpoint("waiNSFWIllustrious_v140.safetensors")
    if lora != "NONE":
        model, clip = LoraLoader().load_lora(model, clip, lora, lorastr, lorastr)

    model, clip = LoraLoader().load_lora(model, clip, "dmd2_sdxl_4step_lora_fp16.safetensors", 1.0, 1.0)

    positive = CLIPTextEncode().encode(clip, positive)[0]
    negative = CLIPTextEncode().encode(clip, negative)[0]
    latent = EmptyLatentImage().generate(width, height, 1)[0]

    with torch.no_grad():
        denoised = KSampler().sample(model, random.randint(0, sys.maxsize), step, cfg, sampler, scheduler, positive, negative, latent, 1.0)[0]
        images = VAEDecode().decode(vae, denoised)[0]
    results = SaveImage().save_images(images)
    return os.path.join(folder_paths.get_output_directory(), results['ui']['images'][0]['filename'])

def main():
    positive = gradio.TextArea(label = "ポジティブプロンプト")
    negative = gradio.TextArea(label = "ネガティブプロンプト")
    width    = gradio.Number(label = "幅", step = 8, value = 1024)
    height   = gradio.Number(label = "高さ", step = 8, value = 1024)
    step     = gradio.Number(label = "ステップ数", value = 8)
    cfg      = gradio.Number(label = "プロンプトの強さ(CFG)", value = 1.0)
    sampler  = gradio.Radio(label = "サンプラー", choices = ["lcm", "euler"], value = "lcm")
    scheduler  = gradio.Radio(label = "スケジューラ", choices = ["simple", "normal"], value = "simple")

    dir_path = folder_paths.folder_names_and_paths["loras"][0][0]
    lora_files = [
      f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]
    print(lora_files)
    lora_files.insert(0, "NONE")

    lora    = gradio.Radio(label = "LoRA", value = "NONE", choices = lora_files)
    lorastr  = gradio.Number(label = "LoRA強度", value = 1.0)

    app = gradio.Interface(
      fn = generate, 
      title = "画像生成", 
      inputs = [positive, negative, width, height, step, cfg, sampler, scheduler, lora, lorastr],
      outputs = "image", submit_btn = "生成", clear_btn = None, flagging_mode = "never")

    app.launch(inbrowser = True)

if __name__ == "__main__":
    main()
