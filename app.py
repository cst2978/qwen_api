import io
import os
import zipfile
from typing import List, Optional, Tuple

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from dotenv import load_dotenv
from huggingface_hub import login

# Model and adapters
HF_BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_ANGLES = "dx8152/Qwen-Edit-2509-Multiple-angles"
LORA_LIGHTNING = "lightx2v/Qwen-Image-Lightning"

# Angle macros (Chinese phrases escaped for ASCII file safety)
ANGLE_MACROS = {
    "Wide-angle": "\\u5e7f\\u89d2",
    "Close-up": "\\u7279\\u5199",
    "Forward": "\\u524d\\u89c6\\u89d2",
    "Left": "\\u5de6\\u89c6\\u89d2",
    "Right": "\\u53f3\\u89c6\\u89d2",
    "Down": "\\u4fef\\u89c6\\u89d2",
    "Rotate 45 deg Left": "\\u5de6\\u8f6c45\\u5ea6",
    "Rotate 45 deg Right": "\\u53f3\\u8f6c45\\u5ea6",
    "Top-down": "\\u4fef\\u62cd",
}

ASPECT_RATIOS = {
    "1:1 (Square)": (1024, 1024),
    "4:3 (Standard)": (1024, 768),
    "3:4 (Portrait)": (768, 1024),
    "16:9 (Widescreen)": (1024, 576),
    "9:16 (Mobile)": (576, 1024),
    "3:2 (Photo)": (1024, 683),
    "2:3 (Portrait Photo)": (683, 1024),
}

app = FastAPI(title="Product Shot Booster API")


def get_gpu_config():
    if not torch.cuda.is_available():
        return {"device": "cpu", "dtype": torch.float32, "enable_lightning": False}
    return {"device": "cuda", "dtype": torch.bfloat16, "enable_lightning": True}


def compose_prompt(angle_phrase: str, bg_text: Optional[str], custom_scene: str, extra_style: str) -> str:
    parts = [angle_phrase]
    if bg_text:
        parts.append(bg_text)
    if custom_scene.strip():
        parts.append(custom_scene.strip())
    if extra_style.strip():
        parts.append(extra_style.strip())
    return " | ".join(parts)


def resize_image(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def make_zip(images: List[Image.Image]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.txt", "Product Shot Booster Export\n")
        for idx, im in enumerate(images):
            img_buf = io.BytesIO()
            im.save(img_buf, format="PNG")
            zf.writestr(f"angle_{idx+1:03d}.png", img_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()


def load_pipeline():
    # Load token from environment variables or hf.env/.env
    load_dotenv("hf.env")
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HUGGINGFACE_HUB_TOKEN is required to download the model.")
    login(token=hf_token, add_to_git_credential=False, new_session=False)

    gpu = get_gpu_config()
    pipe = QwenImageEditPlusPipeline.from_pretrained(HF_BASE_MODEL, torch_dtype=gpu["dtype"])
    pipe = pipe.to(gpu["device"])
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.load_lora_weights(LORA_ANGLES, adapter_name="angles")
    pipe.set_adapters(["angles"], adapter_weights=[1.0])
    return pipe, gpu


pipe, gpu_config = load_pipeline()


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    angles: str = Form("Wide-angle,Close-up,Rotate 45 deg Left,Rotate 45 deg Right,Top-down"),
    aspect_ratio: str = Form("1:1 (Square)"),
    background: str = Form("(None)"),
    custom_scene: str = Form(""),
    extra_style: str = Form("studio-grade lighting, high clarity, ecommerce-ready composition"),
    use_lightning: bool = Form(False),
    seed: int = Form(123),
    steps: int = Form(28),
    guidance_scale: float = Form(1.0),
    true_cfg_scale: float = Form(4.0),
    images_per_prompt: int = Form(1),
):
    if aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail="Invalid aspect_ratio")
    angle_keys = [a.strip() for a in angles.split(",") if a.strip()]
    if not angle_keys:
        raise HTTPException(status_code=400, detail="At least one angle is required")

    try:
        bytes_data = await file.read()
        source_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    target_size = ASPECT_RATIOS[aspect_ratio]
    img = resize_image(source_img, target_size)
    bg_text = None if background == "(None)" else background
    generator = torch.manual_seed(seed)

    pipe.set_adapters(["angles"], adapter_weights=[1.0])
    if use_lightning and gpu_config["enable_lightning"]:
        pipe.load_lora_weights(LORA_LIGHTNING, adapter_name="lightning")
        pipe.set_adapters(["angles", "lightning"], adapter_weights=[1.0, 1.0])

    results = []
    for angle in angle_keys:
        if angle not in ANGLE_MACROS:
            raise HTTPException(status_code=400, detail=f"Invalid angle: {angle}")
        full_prompt = compose_prompt(ANGLE_MACROS[angle], bg_text, custom_scene, extra_style)
        with torch.inference_mode():
            out = pipe(
                image=[img],
                prompt=full_prompt,
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=" ",
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=images_per_prompt,
                height=target_size[1],
                width=target_size[0],
            )
        results.extend(out.images)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    zip_bytes = make_zip(results)
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=product_shot_booster.zip"},
    )
