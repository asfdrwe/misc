import gradio
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ComfyUI'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ComfyUI/comfy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ComfyUI/comfy_extra'))

import main, folder_paths

from ComfyUI.nodes import CLIPTextEncode, VAEDecode, KSampler, VAELoader, LoraLoaderModelOnly

from ComfyUI.comfy_extras.nodes_wan import WanVaceToVideo
from ComfyUI.comfy_extras.nodes_video import SaveWEBM

import importlib
comfyui_gguf = importlib.import_module("ComfyUI.custom_nodes.comfyui-gguf.nodes")

import torch
import numpy as np
import random
from PIL import Image
import torch

def generate(image, positive, height, width, length, lora, lorastr):
    pixels = np.array(image).astype(np.float32) / 255.0
    pixels = torch.from_numpy(pixels)[None,]

    clip = comfyui_gguf.CLIPLoaderGGUF().load_clip("WanVideo/umt5-xxl-encoder-Q5_K_M.gguf", "wan")[0]

    positive = CLIPTextEncode().encode(clip, positive)[0]
    negative = CLIPTextEncode().encode(clip, "")[0]

    vae = VAELoader().load_vae("WanVideo/wan_2.1_vae.safetensors")[0]

    positive, negative, latent = WanVaceToVideo().encode(positive, negative, vae, width, height, length, 1, 1.0, reference_image=pixels)

    model = comfyui_gguf.UnetLoaderGGUF().load_unet("WanVideo/Wan2.1-VACE-14B-Q4_K_S.gguf")[0]
    if lora != "NONE":
        model = LoraLoaderModelOnly().load_lora_model_only(model, os.path.join("WanVideo", lora), lorastr)[0]
    model = LoraLoaderModelOnly().load_lora_model_only(model, "WanVideo/Wan21_CausVid_14B_T2V_lora_rank32_v1_5_no_first_block.safetensors", 1.0)[0]

    with torch.no_grad():
        denoised = KSampler().sample(model, random.randint(0, sys.maxsize), 6, 1.0, "uni_pc", "simple", positive, negative, latent, 1.0)[0]
        images = VAEDecode().decode(vae, denoised)[0]
    results = SaveWEBM().save_images(images, "vp9", 16.0, "ComfyUI", 32)
    return os.path.join(folder_paths.get_output_directory(), results['ui']['images'][0]['filename'])


def main():
    image    = gradio.Image(type='pil', label = "入力画像")
    positive = gradio.TextArea(label = "ポジティブプロンプト")
    height   = gradio.Number(label = "高さ", value = 640)
    width    = gradio.Number(label = "幅", value = 480)
    length   = gradio.Number(label = "長さ", value = 17)
    video    = gradio.Video(autoplay = True, loop = True)

    dir_path = os.path.join(folder_paths.folder_names_and_paths["loras"][0][0],"WanVideo")
    lora_files = [
      f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]
    print(lora_files)
    lora_files.insert(0, "NONE")

    lora    = gradio.Radio(label = "LoRA", value = "NONE", choices = lora_files)
    lorastr  = gradio.Number(label = "LoRA強度", value = 1.0)

    app = gradio.Interface(
      fn = generate, 
      title = "動画生成", 
      inputs = [image, positive, height, width, length, lora, lorastr],
      outputs = [video], 
      submit_btn = "生成", 
      clear_btn = None, 
      flagging_mode = "never"
    )

    app.launch(inbrowser = True)

if __name__ == "__main__":
    main()
