import os
import json
import base64
import requests
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import cv2  # NEW: for robust image decoding (WebP, etc.)
import random  # NEW: optional local seeding
import time  # NEW: for retry backoff
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout  # NEW: request timeout

p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(p, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# NEW: small helpers for debug formatting
def _fmt_ms(seconds):
    return f"{seconds * 1000:.1f} ms"

def _safe_len(b):
    try:
        return len(b)
    except Exception:
        return 0

# NEW: robust decoder for various response encodings (raw bytes, webp, base64-wrapped, data URLs)
def _decode_image_to_numpy(img_bytes, mime_hint=""):
    # 1) Try PIL directly
    try:
        im = Image.open(BytesIO(img_bytes))
        im.load()
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.array(im).astype(np.float32) / 255.0
    except Exception:
        pass

    # 2) Try OpenCV directly
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        mat = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if mat is not None:
            if mat.ndim == 2:
                mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2RGB)
            elif mat.shape[2] == 4:
                mat = cv2.cvtColor(mat, cv2.COLOR_BGRA2RGB)
            else:
                mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
            return mat.astype(np.float32) / 255.0
    except Exception:
        pass

    # 3) If the bytes actually contain ASCII/base64 content (or data URL), try base64 decode
    try:
        s = img_bytes.decode("ascii", errors="ignore").strip()
        if s.startswith("data:"):
            comma = s.find(",")
            if comma != -1:
                s = s[comma + 1:]
        s_clean = "".join(s.split())
        decoded = base64.b64decode(s_clean, validate=False)

        # Retry PIL
        try:
            im = Image.open(BytesIO(decoded))
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
            return np.array(im).astype(np.float32) / 255.0
        except Exception:
            pass

        # Retry OpenCV
        try:
            arr = np.frombuffer(decoded, dtype=np.uint8)
            mat = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if mat is not None:
                if mat.ndim == 2:
                    mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2RGB)
                elif mat.shape[2] == 4:
                    mat = cv2.cvtColor(mat, cv2.COLOR_BGRA2RGB)
                else:
                    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
                return mat.astype(np.float32) / 255.0
        except Exception:
            pass
    except Exception:
        pass

    raise ValueError(f"Failed to decode image (mime={mime_hint}) with PIL/ OpenCV/ base64.")

class ComfyUI_NanoBanana:
    def __init__(self, api_key=None):
        env_key = os.environ.get("GEMINI_API_KEY")
        
        # Common placeholder values to ignore
        placeholders = {"token_here", "place_token_here", "your_api_key",
                        "api_key_here", "enter_your_key", "<api_key>"}

        if env_key and env_key.lower().strip() not in placeholders:
            self.api_key = env_key
        else:
            self.api_key = api_key
            if self.api_key is None:
                config = get_config()
                self.api_key = config.get("GEMINI_API_KEY")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Generate a high-quality, photorealistic image", 
                    "multiline": True,
                    "tooltip": "Describe what you want to generate or edit"
                }),
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {
                    "default": "generate",
                    "tooltip": "Choose the type of image operation"
                }),
            },
            "optional": {
                "reference_image_1": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Primary reference image for editing/style transfer"
                }),
                "reference_image_2": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Second reference image (optional)"
                }),
                "reference_image_3": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Third reference image (optional)"
                }),
                "reference_image_4": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Fourth reference image (optional)"
                }),
                "reference_image_5": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Fifth reference image (optional)"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your Gemini API key (paid tier required)"
                }),
                "batch_count": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4, 
                    "step": 1,
                    "tooltip": "Number of images to generate (costs multiply)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.1,
                    "tooltip": "Creativity level (0.0 = deterministic, 1.0 = very creative)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9007199254740991,
                    "step": 1,
                    "tooltip": "Random seed (-1 = auto). Large seeds will be normalized to 32-bit for determinism."
                }),
                "quality": (["standard", "high"], {
                    "default": "high",
                    "tooltip": "Image generation quality"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "1:1",
                    "tooltip": "Output image aspect ratio"
                }),
                "character_consistency": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Maintain character consistency across edits"
                }),

                "enable_safety": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable content safety filters"
                }),
                # NEW: debug toggle
                "debug_logging": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print detailed timing and payload info for bottleneck analysis"
                }),
                # NEW: per-attempt request timeout (seconds)
                "request_timeout": ("FLOAT", {
                    "default": 60.0,
                    "min": 5.0,
                    "max": 600.0,
                    "step": 5.0,
                    "tooltip": "Per-attempt API request timeout seconds (timeout -> retry)."
                }),
                # NEW: timeout handling strategy
                "timeout_strategy": (["poll", "future"], {
                    "default": "poll",
                    "tooltip": "poll: cooperative polling (non-blocking). future: use future.result(timeout=...)."
                }),
                # NEW: overall hard stop (seconds) for whole API call (0 = disabled)
                "hard_overall_timeout": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 10.0,
                    "tooltip": "Abort entire call_nano_banana_api after this many seconds (0=off)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "operation_log")
    FUNCTION = "nano_banana_generate"
    CATEGORY = "Nano Banana (Gemini 2.5 Flash Image)"
    DESCRIPTION = "Generate and edit images using Google's Nano Banana (Gemini 2.5 Flash Image) model. Requires paid API access."

    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor.cpu()
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor[0]
        
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        return Image.fromarray(image_np, mode='RGB')

    def resize_image(self, image, max_size=2048):
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            return image.resize((new_width, new_height), Image.LANCZOS)
        return image

    def create_placeholder_image(self, width=512, height=512):
        """Create a placeholder image when generation fails"""
        img = Image.new('RGB', (width, height), color=(100, 100, 100))
        # Add text overlay indicating error
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            draw.text((width//2-50, height//2), "Generation\nFailed", fill=(255, 255, 255))
        except:
            pass
        
        image_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_array).unsqueeze(0)

    def prepare_images_for_api(self, img1=None, img2=None, img3=None, img4=None, img5=None):
        """Convert up to 5 tensor images to raw PNG bytes for API (resized ≤2048px)"""
        encoded_images = []
        for i, img in enumerate([img1, img2, img3, img4, img5], 1):
            if img is None:
                continue
            if isinstance(img, torch.Tensor):
                if len(img.shape) == 4:
                    pil_image = self.tensor_to_image(img[0])
                else:
                    pil_image = self.tensor_to_image(img)
                # Resize to keep payload/API happy
                pil_image = self.resize_image(pil_image, max_size=2048)
                encoded_images.append(self._image_to_base64(pil_image))
        return encoded_images

    def _image_to_base64(self, pil_image):
        """Convert PIL image to raw PNG bytes payload expected by SDK"""
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        # Return bytes directly; SDK will wrap as Blob
        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": img_bytes,  # bytes, not base64 string
            }
        }

    def build_prompt_for_operation(self, prompt, operation, has_references=False, aspect_ratio="1:1", character_consistency=True):
        """Build optimized prompt based on operation type"""
        
        aspect_instructions = {
            "1:1": "square format",
            "16:9": "widescreen landscape format",
            "9:16": "portrait format",
            "4:3": "standard landscape format", 
            "3:4": "standard portrait format"
        }
        
        base_quality = "Generate a high-quality, photorealistic image"
        format_instruction = f"in {aspect_instructions.get(aspect_ratio, 'square format')}"
        
        if operation == "generate":
            if has_references:
                final_prompt = f"{base_quality} inspired by the style and elements of the reference images. {prompt}. {format_instruction}."
            else:
                final_prompt = f"{base_quality} of: {prompt}. {format_instruction}."
                
        elif operation == "edit":
            if not has_references:
                return "Error: Edit operation requires reference images"
            # No aspect ratio for edit - preserve original image dimensions
            final_prompt = f"Edit the provided reference image(s). {prompt}. Maintain the original composition and quality while making the requested changes."
            
        elif operation == "style_transfer":
            if not has_references:
                return "Error: Style transfer requires reference images"
            final_prompt = f"Apply the style from the reference images to create: {prompt}. Blend the stylistic elements naturally. {format_instruction}."
            
        elif operation == "object_insertion":
            if not has_references:
                return "Error: Object insertion requires reference images"
            final_prompt = f"Insert or blend the following into the reference image(s): {prompt}. Ensure natural lighting, shadows, and perspective. {format_instruction}."
        
        if character_consistency and has_references:
            final_prompt += " Maintain character consistency and visual identity from the reference images."
            
        return final_prompt

    def _normalize_seed(self, seed):
        """Normalize any large seed into 32-bit non-negative range"""
        try:
            s = int(seed)
        except Exception:
            return None, None
        if s < 0:
            return None, None
        max32 = (1 << 31) - 1  # 2147483647
        norm = s % (max32 + 1)  # -> [0, 2147483647]
        return s, norm

    # NEW: add request_timeout param
    def call_nano_banana_api(self, prompt, encoded_images, temperature, batch_count, enable_safety,
                             seed=None, retries=3, debug_logging=False, request_timeout=60.0,
                             timeout_strategy="poll", hard_overall_timeout=0.0):
        """Make API call to Gemini 2.5 Flash Image using the working v6 approach with retries per batch"""
        overall_t0 = time.perf_counter()
        pre_debug_lines = []
        operation_log = ""  # moved before emit

        # NEW: immediate logger
        def emit(msg):
            nonlocal operation_log
            operation_log += msg if msg.endswith("\n") else (msg + "\n")
            if debug_logging:
                try:
                    print(msg, end="" if msg.endswith("\n") else "\n", flush=True)
                except Exception:
                    pass

        try:
            sdk_t0 = time.perf_counter()
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key)
            pre_debug_lines.append(f"SDK client init: {_fmt_ms(time.perf_counter() - sdk_t0)}")

            # NEW: try to pass seed if supported by SDK/model
            seed_applied = False
            cfg_t0 = time.perf_counter()
            try:
                generation_config = types.GenerateContentConfig(
                    temperature=temperature,
                    response_modalities=['Text', 'Image'],
                    seed=seed if (seed is not None and seed >= 0) else None
                )
                if seed is not None and seed >= 0:
                    seed_applied = True
            except TypeError:
                generation_config = types.GenerateContentConfig(
                    temperature=temperature,
                    response_modalities=['Text', 'Image']
                )
            pre_debug_lines.append(f"Build generation config: {_fmt_ms(time.perf_counter() - cfg_t0)}")

            # Build parts with text and image bytes
            parts_t0 = time.perf_counter()
            parts = [types.Part(text=prompt)]
            total_input_image_bytes = 0
            for img_data in encoded_images:
                # Be robust if data was accidentally base64-encoded
                data_field = img_data.get("inline_data", {}).get("data", b"")
                if isinstance(data_field, str):
                    try:
                        data_bytes = base64.b64decode(data_field)
                    except Exception:
                        continue
                else:
                    data_bytes = data_field
                total_input_image_bytes += _safe_len(data_bytes)
                mime = img_data.get("inline_data", {}).get("mime_type", "image/png")
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime,
                            data=data_bytes
                        )
                    )
                )
            contents = [types.Content(role="user", parts=parts)]
            pre_debug_lines.append(
                f"Build parts: {_fmt_ms(time.perf_counter() - parts_t0)} "
                f"(prompt_len={len(prompt)}, ref_images={len(encoded_images)}, payload≈{total_input_image_bytes/1024:.1f} KB)"
            )

            if debug_logging and pre_debug_lines:
                emit("DEBUG TIMINGS (SDK/build):")
                for line in pre_debug_lines:
                    emit(line)

            # NEW: seed status note
            if seed is not None and seed >= 0:
                emit(f"Seed requested: {seed}")
                if not seed_applied:
                    emit("Seed parameter not supported by current SDK/model; determinism not guaranteed.")

            all_generated_images = []
            overall_deadline = (overall_t0 + hard_overall_timeout) if hard_overall_timeout > 0 else None

            for i in range(batch_count):
                batch_images = []
                response_text = ""
                last_error = None
                for attempt in range(1, int(max(1, retries)) + 1):
                    if overall_deadline and time.perf_counter() > overall_deadline:
                        emit(f"Overall hard timeout ({hard_overall_timeout:.1f}s) hit before batch {i+1} attempt {attempt}.")
                        last_error = RuntimeError("Overall hard timeout exceeded.")
                        break

                    timeout_hit = False
                    req_t0 = time.perf_counter()

                    try:
                        def _do_request():
                            return client.models.generate_content(
                                model="gemini-2.5-flash-image-preview",
                                contents=contents,
                                config=generation_config
                            )

                        if timeout_strategy == "future":
                            # OLD style (may block if underlying call ignores cancellation)
                            with ThreadPoolExecutor(max_workers=1) as ex:
                                future = ex.submit(_do_request)
                                try:
                                    response = future.result(timeout=request_timeout)
                                except _FuturesTimeout:
                                    timeout_hit = True
                                    future.cancel()
                                    raise RuntimeError(f"Request timed out after {request_timeout:.1f}s (future strategy)")
                        else:
                            # NEW polling strategy to avoid blocking shutdown wait
                            ex = ThreadPoolExecutor(max_workers=1)
                            future = ex.submit(_do_request)
                            deadline = time.time() + request_timeout
                            poll_interval = 0.25
                            try:
                                while True:
                                    if future.done():
                                        response = future.result()
                                        break
                                    if time.time() > deadline:
                                        timeout_hit = True
                                        emit(f"Attempt timed out at {request_timeout:.1f}s (poll strategy) – proceeding to retry.")
                                        # Do not block on unfinished thread; allow it to linger
                                        raise RuntimeError(f"Request timed out after {request_timeout:.1f}s (poll strategy)")
                                    time.sleep(poll_interval)
                            finally:
                                # Non-blocking shutdown if still running
                                try:
                                    if future.done():
                                        ex.shutdown(wait=True)
                                    else:
                                        ex.shutdown(wait=False)
                                except Exception:
                                    pass

                        req_ms = (time.perf_counter() - req_t0) * 1000.0

                        # ...existing parse logic...
                        parse_t0 = time.perf_counter()
                        if hasattr(response, 'candidates') and response.candidates:
                            for candidate in response.candidates:
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            response_text += part.text + "\n"
                                        if hasattr(part, 'inline_data') and part.inline_data:
                                            try:
                                                mime = getattr(part.inline_data, "mime_type", None) or "image/png"
                                                image_binary = part.inline_data.data
                                                if isinstance(image_binary, str):
                                                    try:
                                                        image_binary = base64.b64decode(image_binary)
                                                    except Exception:
                                                        image_binary = b""
                                                if isinstance(image_binary, (bytes, bytearray)) and len(image_binary) > 0:
                                                    batch_images.append({"data": bytes(image_binary), "mime": mime})
                                            except Exception as img_error:
                                                emit(f"Error extracting image: {str(img_error)}")
                        parse_ms = (time.perf_counter() - parse_t0) * 1000.0

                        if debug_logging:
                            emit(f"Batch {i+1} attempt {attempt}: request {req_ms:.1f} ms, parse {parse_ms:.1f} ms, images={len(batch_images)}, text_len={len(response_text)}")
                        break

                    except Exception as e:
                        dur_ms = (time.perf_counter() - req_t0) * 1000.0
                        kind = "timeout" if timeout_hit else "error"
                        last_error = e
                        if attempt < retries and (not (overall_deadline and time.perf_counter() > overall_deadline)):
                            emit(f"Batch {i+1} attempt {attempt} {kind} ({dur_ms:.1f} ms): {e}; retrying...")
                            time.sleep(min(2.0, 0.5 * attempt))
                            continue
                        else:
                            emit(f"Batch {i+1} {kind} final fail after {attempt} attempt(s) ({dur_ms:.1f} ms): {e}")
                            break

                if batch_images:
                    all_generated_images.extend(batch_images)
                    emit(f"Batch {i+1}: Generated {len(batch_images)} images")
                else:
                    if last_error and "timeout" in str(last_error).lower():
                        emit(f"Batch {i+1}: Aborted due to timeout.")
                    else:
                        snippet = (response_text[:100] + "...") if response_text else ""
                        emit(f"Batch {i+1}: No images. {('Last error: ' + str(last_error)) if last_error else 'Text: ' + snippet}")

                if overall_deadline and time.perf_counter() > overall_deadline:
                    emit(f"Stopping further batches due to overall hard timeout ({hard_overall_timeout:.1f}s).")
                    break

            generated_tensors = []
            decode_total_t0 = time.perf_counter()
            if all_generated_images:
                for idx, item in enumerate(all_generated_images, 1):
                    try:
                        # item may be dict with data/mime or raw bytes (legacy)
                        if isinstance(item, dict):
                            img_bytes = item.get("data", b"")
                            mime = item.get("mime", "application/octet-stream")
                        else:
                            img_bytes = item
                            mime = "application/octet-stream"

                        # Use robust decoder
                        dec_t0 = time.perf_counter()
                        img_np = _decode_image_to_numpy(img_bytes, mime_hint=mime)
                        dec_ms = (time.perf_counter() - dec_t0) * 1000.0

                        h, w = (img_np.shape[0], img_np.shape[1]) if img_np.ndim >= 2 else (0, 0)
                        if debug_logging:
                            emit(f"Decode image {idx}: {mime}, bytes={_safe_len(img_bytes)} -> {w}x{h}, {_fmt_ms(dec_ms/1000.0)}")

                        img_tensor = torch.from_numpy(img_np)[None,]
                        generated_tensors.append(img_tensor)
                    except Exception as e:
                        # Log a short hex prefix for debugging
                        hex_head = img_bytes[:12].hex() if isinstance(img_bytes, (bytes, bytearray)) else "n/a"
                        emit(f"Error processing image: {e} head={hex_head}")

            if debug_logging:
                emit(f"Decode all images: {_fmt_ms(time.perf_counter() - decode_total_t0)}")
                emit(f"Total (call_nano_banana_api): {_fmt_ms(time.perf_counter() - overall_t0)}")

            # NEW: if nothing generated, raise instead of returning empty
            if not generated_tensors:
                raise RuntimeError(f"Google image generation returned no images.\n{operation_log.strip()}")

            return generated_tensors, operation_log

        except ImportError as ie:
            # NEW: bubble up ImportError
            raise ImportError("google.genai not available. Please install google-genai and ensure paid API access.") from ie
        except Exception as e:
            # NEW: bubble up other errors
            raise RuntimeError(f"Error in v6 method: {str(e)}") from e

    def nano_banana_generate(self, prompt, operation, reference_image_1=None, reference_image_2=None, 
                           reference_image_3=None, reference_image_4=None, reference_image_5=None, api_key="", 
                           batch_count=1, temperature=0.7, quality="high", aspect_ratio="1:1",
                           character_consistency=True, enable_safety=True, seed=-1, retries=3, debug_logging=True, request_timeout=60.0, timeout_strategy="poll", hard_overall_timeout=0.0):
        outer_t0 = time.perf_counter()
        stage_marks = []
        def _mark(label, tstore=[time.perf_counter()]):
            now = time.perf_counter()
            stage_marks.append((label, now - tstore[0]))
            tstore[0] = now

        # NEW: immediate logger
        operation_log = ""
        def emit(msg):
            nonlocal operation_log
            operation_log += msg if msg.endswith("\n") else (msg + "\n")
            if debug_logging:
                try:
                    print(msg, end="" if msg.endswith("\n") else "\n", flush=True)
                except Exception:
                    pass

        # Validate and set API key
        if api_key.strip():
            self.api_key = api_key
            save_config({"GEMINI_API_KEY": self.api_key})
        _mark("API key validation")

        if not self.api_key:
            error_msg = "NANO BANANA ERROR: No API key provided!\n\n"
            error_msg += "Gemini 2.5 Flash Image requires a PAID API key.\n"
            error_msg += "Get yours at: https://aistudio.google.com/app/apikey\n"
            error_msg += "Note: Free tier users cannot access image generation models."
            return (self.create_placeholder_image(), error_msg)

        try:
            # NEW: normalize seed and apply locally (server may ignore)
            req_seed, norm_seed = self._normalize_seed(seed)
            if norm_seed is not None:
                try:
                    random.seed(norm_seed)
                    np.random.seed(norm_seed)
                    torch.manual_seed(norm_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(norm_seed)
                except Exception:
                    pass
            _mark("Seed normalization/setup")

            # Process reference images (up to 5)
            ref_shapes = []
            for idx, ref in enumerate([reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5], 1):
                if isinstance(ref, torch.Tensor):
                    # Accept (B,H,W,C) or (H,W,C)
                    shape = tuple(ref.shape)
                    ref_shapes.append((idx, shape))
            encoded_images = self.prepare_images_for_api(
                reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5
            )
            enc_bytes = sum(_safe_len(e.get("inline_data", {}).get("data", b"")) for e in encoded_images)
            has_references = len(encoded_images) > 0
            _mark("Encode reference images")

            # Build optimized prompt
            final_prompt = self.build_prompt_for_operation(
                prompt, operation, has_references, aspect_ratio, character_consistency
            )
            _mark("Build prompt")
            
            if "Error:" in final_prompt:
                return (self.create_placeholder_image(), final_prompt)
            
            # Add quality instructions
            if quality == "high":
                final_prompt += " Use the highest quality settings available."
            _mark("Attach quality")

            # Log operation start (print header immediately)
            emit("NANO BANANA OPERATION LOG")
            # NEW: Env and hardware info
            if debug_logging:
                try:
                    cuda = torch.cuda.is_available()
                    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU"
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda else 0
                    vram_reserved = torch.cuda.memory_reserved(0) / (1024**3) if cuda else 0
                    vram_alloc = torch.cuda.memory_allocated(0) / (1024**3) if cuda else 0
                except Exception:
                    cuda, gpu_name, vram_total, vram_reserved, vram_alloc = False, "Unknown", 0, 0, 0
                emit("DEBUG ENV:")
                emit(f"- CPU cores: {os.cpu_count()}")
                emit(f"- Torch: {torch.__version__}")
                emit(f"- CUDA available: {cuda}, Device: {gpu_name}, VRAM total≈{vram_total:.2f} GB, reserved≈{vram_reserved:.2f} GB, alloc≈{vram_alloc:.2f} GB")
                if ref_shapes:
                    emit(f"- Reference tensor shapes: {ref_shapes}")

            emit(f"Operation: {operation.upper()}")
            emit(f"Reference Images: {len(encoded_images)} (payload≈{enc_bytes/1024:.1f} KB)")
            emit(f"Batch Count: {batch_count}")
            emit(f"Temperature: {temperature}")
            emit(f"Seed: {req_seed if (req_seed is not None) else 'auto'}")
            if req_seed is not None and req_seed != norm_seed:
                emit(f"Normalized seed (32-bit): {norm_seed}")
            emit(f"Retries: {int(max(1, retries))}")
            emit(f"Quality: {quality}")
            emit(f"Aspect Ratio: {aspect_ratio}")
            emit(f"Character Consistency: {character_consistency}")
            emit(f"Safety Filters: {enable_safety}")
            emit(f"Request Timeout (per attempt): {request_timeout:.1f}s")
            emit(f"Timeout Strategy: {timeout_strategy}")
            if hard_overall_timeout > 0:
                emit(f"Overall Hard Timeout: {hard_overall_timeout:.1f}s")
            emit("Note: Output resolution determined by API (max ~1024px)")
            emit(f"Prompt (len={len(final_prompt)}): {final_prompt[:150]}...\n")

            # Pre-API debug timings (outer setup)
            if debug_logging and stage_marks:
                emit("DEBUG TIMINGS (outer pre-API):")
                for i, (label, secs) in enumerate(stage_marks, 1):
                    emit(f"{i:02d}. {label}: {_fmt_ms(secs)}")
                emit("")
            stage_marks.clear()

            # Make API call with normalized seed and retries
            api_t0 = time.perf_counter()
            generated_images, api_log = self.call_nano_banana_api(
                final_prompt, encoded_images, temperature, batch_count, enable_safety,
                seed=norm_seed, retries=int(max(1, retries)), debug_logging=debug_logging,
                request_timeout=request_timeout, timeout_strategy=timeout_strategy,
                hard_overall_timeout=hard_overall_timeout
            )
            api_secs = time.perf_counter() - api_t0
            # api_log already printed via emit inside call; still append the returned log
            operation_log += api_log
            _mark("API call")

            # Process results
            post_t0 = time.perf_counter()
            if generated_images:
                # Combine all generated images into a batch tensor
                combined_tensor = torch.cat(generated_images, dim=0)
                post_secs = time.perf_counter() - post_t0
                _mark("Concat tensors")

                # Calculate approximate cost
                approx_cost = len(generated_images) * 0.039  # ~$0.039 per image
                emit(f"\nEstimated cost: ~${approx_cost:.3f}")
                emit(f"Successfully generated {len(generated_images)} image(s)!")

                # Tail debug timings (outer, incl. API call & concat)
                if debug_logging and stage_marks:
                    emit("DEBUG TIMINGS (outer post-API):")
                    for i, (label, secs) in enumerate(stage_marks, 1):
                        emit(f"{i:02d}. {label}: {_fmt_ms(secs)}")
                    emit(f"Total (outer wrapper): {_fmt_ms((time.perf_counter() - outer_t0))}")
                return (combined_tensor, operation_log)
            else:
                post_secs = time.perf_counter() - post_t0
                _mark("Post-process (no images)")
                emit("\nNo images were generated. Check the log above for details.")
                if debug_logging and stage_marks:
                    emit("DEBUG TIMINGS (outer post-API):")
                    for i, (label, secs) in enumerate(stage_marks, 1):
                        emit(f"{i:02d}. {label}: {_fmt_ms(secs)}")
                    emit(f"Total (outer wrapper): {_fmt_ms((time.perf_counter() - outer_t0))}")
                return (self.create_placeholder_image(), operation_log)
                
        except Exception:
            # NEW: re-raise so ComfyUI surfaces the error directly
            raise

# Node registration
NODE_CLASS_MAPPINGS = {
    "ComfyUI_NanoBanana": ComfyUI_NanoBanana,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_NanoBanana": "Nano Banana (Gemini 2.5 Flash Image)",
}