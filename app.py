import io
import os
import zipfile
from typing import List, Tuple, Optional

import streamlit as st
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from dotenv import load_dotenv
from huggingface_hub import login

HF_BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_ANGLES = "dx8152/Qwen-Edit-2509-Multiple-angles"
LORA_LIGHTNING = "lightx2v/Qwen-Image-Lightning"

ANGLE_MACROS = {
    "Wide-angle": "\u5e7f\u89d2",
    "Close-up": "\u7279\u5199",
    "Forward": "\u524d\u89c6\u89d2",
    "Left": "\u5de6\u89c6\u89d2",
    "Right": "\u53f3\u89c6\u89d2",
    "Down": "\u4fef\u89c6\u89d2",
    "Rotate 45 deg Left": "\u5de6\u8f6c45\u5ea6",
    "Rotate 45 deg Right": "\u53f3\u8f6c45\u5ea6",
    "Top-down": "\u4fef\u62cd",
}

BACKGROUND_PRESETS = {
    "(None)": None,
    "Pure Studio (white seamless)": "in a professional studio with seamless white background, soft shadows, product centered",
    "Soft Gray Studio": "in a professional studio with seamless soft gray background, gentle vignette, softbox lighting",
    "Lifestyle (cozy desk)": "on a cozy wooden desk near a window, soft natural light, minimal props",
    "Lifestyle (marble)": "on a clean white marble surface, bright daylight, subtle reflections",
    "Lifestyle (outdoor)": "outdoors on a neutral table, soft shade, bokeh background",
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


@st.cache_resource(show_spinner=False)
def load_pipe(device: str, dtype):
    pipe = QwenImageEditPlusPipeline.from_pretrained(HF_BASE_MODEL, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.load_lora_weights(LORA_ANGLES, adapter_name="angles")
    pipe.set_adapters(["angles"], adapter_weights=[1.0])
    return pipe


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


def generate_images(pipe, source_img: Image.Image, angle_keys, bg_key: str, custom_scene: str,
                    extra_style: str, aspect_ratio: str, use_lightning: bool, seed: int, steps: int,
                    guidance_scale: float, true_cfg_scale: float, images_per_prompt: int):
    target_size = ASPECT_RATIOS[aspect_ratio]
    img = resize_image(source_img, target_size)
    bg_text = BACKGROUND_PRESETS.get(bg_key)
    angle_list = angle_keys or []
    generator = torch.manual_seed(seed)
    pipe.set_adapters(["angles"], adapter_weights=[1.0])
    if use_lightning:
        pipe.load_lora_weights(LORA_LIGHTNING, adapter_name="lightning")
        pipe.set_adapters(["angles", "lightning"], adapter_weights=[1.0, 1.0])
    results = []
    for angle in angle_list:
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
    return results


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


def main():
    st.set_page_config(page_title="Product Shot Booster", layout="wide")
    st.title("Product Shot Booster (Qwen-Image-Edit-2509)")

    # Load token from hf.env (preferred) or .env for environments that block dotfiles.
    load_dotenv("hf.env")
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False, new_session=False)

    gpu = get_gpu_config()
    pipe = load_pipe(gpu["device"], gpu["dtype"])

    col_left, col_right = st.columns([1, 2], gap="large")
    with col_left:
        file = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg", "webp"])
        angle_keys = st.multiselect(
            "Camera Angles",
            list(ANGLE_MACROS.keys()),
            default=["Wide-angle", "Close-up", "Rotate 45 deg Left", "Rotate 45 deg Right", "Top-down"],
        )
        aspect_ratio = st.selectbox("Aspect Ratio", list(ASPECT_RATIOS.keys()), index=0)
        bg_key = st.selectbox("Background", list(BACKGROUND_PRESETS.keys()), index=0)
        custom_scene = st.text_area("Custom Scene", "", placeholder="e.g., on a matte black table with soft reflections")
        extra_style = st.text_area("Style Notes", "studio-grade lighting, high clarity, ecommerce-ready composition")
        with st.expander("Advanced"):
            use_lightning = st.checkbox("Fast Mode (Lightning LoRA)", value=gpu["enable_lightning"])
            seed = st.number_input("Seed", value=123, step=1)
            steps = st.slider("Inference Steps", 10, 60, 28, 1)
            guidance_scale = st.slider("Guidance Scale", 0.0, 8.0, 1.0, 0.1)
            true_cfg_scale = st.slider("True CFG Scale", 0.0, 10.0, 4.0, 0.1)
            images_per_prompt = st.slider("Images per Angle", 1, 4, 1, 1)
        generate = st.button("Generate", type="primary")

    with col_right:
        placeholder = st.empty()
        zip_slot = st.empty()

    if generate:
        if not file:
            st.warning("Please upload an image first.")
            return
        if not angle_keys:
            st.warning("Please select at least one angle.")
            return
        source_img = Image.open(file).convert("RGB")
        with st.spinner("Generating..."):
            images = generate_images(
                pipe,
                source_img,
                angle_keys,
                bg_key,
                custom_scene,
                extra_style,
                aspect_ratio,
                use_lightning,
                int(seed),
                int(steps),
                float(guidance_scale),
                float(true_cfg_scale),
                int(images_per_prompt),
            )
        placeholder.image(
            images,
            caption=[f"{i+1}: {angle_keys[i // images_per_prompt]}" for i in range(len(images))],
            use_column_width=True,
        )
        if images:
            zip_bytes = make_zip(images)
            zip_slot.download_button(
                "Download ZIP",
                data=zip_bytes,
                file_name="product_shot_booster.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
