import os
import json
import base64
import requests
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import cv2  # NEW: for robust image decoding (WebP, etc.)

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

    def call_nano_banana_api(self, prompt, encoded_images, temperature, batch_count, enable_safety):
        """Make API call to Gemini 2.5 Flash Image using the working v6 approach"""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                response_modalities=['Text', 'Image']
            )

            # Build parts with text and image bytes
            parts = [types.Part(text=prompt)]
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

            all_generated_images = []
            operation_log = ""

            for i in range(batch_count):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-image-preview",
                        contents=contents,
                        config=generation_config
                    )
                    batch_images = []
                    response_text = ""

                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        response_text += part.text + "\n"
                                    # Image parts are returned as inline_data Blob
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        try:
                                            mime = getattr(part.inline_data, "mime_type", None) or "image/png"
                                            image_binary = part.inline_data.data  # bytes or base64 string
                                            if isinstance(image_binary, str):
                                                try:
                                                    image_binary = base64.b64decode(image_binary)
                                                except Exception:
                                                    image_binary = b""
                                            if isinstance(image_binary, (bytes, bytearray)) and len(image_binary) > 0:
                                                # Keep bytes and mime for robust decoding later
                                                batch_images.append({"data": bytes(image_binary), "mime": mime})
                                        except Exception as img_error:
                                            operation_log += f"Error extracting image: {str(img_error)}\n"

                    if batch_images:
                        all_generated_images.extend(batch_images)
                        operation_log += f"Batch {i+1}: Generated {len(batch_images)} images\n"
                    else:
                        operation_log += f"Batch {i+1}: No images found. Text: {response_text[:100]}...\n"

                except Exception as batch_error:
                    operation_log += f"Batch {i+1} error: {str(batch_error)}\n"

            generated_tensors = []
            if all_generated_images:
                for item in all_generated_images:
                    try:
                        # item may be dict with data/mime or raw bytes (legacy)
                        if isinstance(item, dict):
                            img_bytes = item.get("data", b"")
                            mime = item.get("mime", "application/octet-stream")
                        else:
                            img_bytes = item
                            mime = "application/octet-stream"

                        # Use robust decoder
                        img_np = _decode_image_to_numpy(img_bytes, mime_hint=mime)

                        img_tensor = torch.from_numpy(img_np)[None,]
                        generated_tensors.append(img_tensor)
                    except Exception as e:
                        # Log a short hex prefix for debugging
                        hex_head = img_bytes[:12].hex() if isinstance(img_bytes, (bytes, bytearray)) else "n/a"
                        operation_log += f"Error processing image: {e} head={hex_head}\n"

            return generated_tensors, operation_log

        except ImportError:
            operation_log = "google.genai not available, using requests fallback\n"
            return [], operation_log
        except Exception as e:
            operation_log = f"Error in v6 method: {str(e)}\n"
            return [], operation_log

    def nano_banana_generate(self, prompt, operation, reference_image_1=None, reference_image_2=None, 
                           reference_image_3=None, reference_image_4=None, reference_image_5=None, api_key="", 
                           batch_count=1, temperature=0.7, quality="high", aspect_ratio="1:1",
                           character_consistency=True, enable_safety=True):
        
        # Validate and set API key
        if api_key.strip():
            self.api_key = api_key
            save_config({"GEMINI_API_KEY": self.api_key})

        if not self.api_key:
            error_msg = "NANO BANANA ERROR: No API key provided!\n\n"
            error_msg += "Gemini 2.5 Flash Image requires a PAID API key.\n"
            error_msg += "Get yours at: https://aistudio.google.com/app/apikey\n"
            error_msg += "Note: Free tier users cannot access image generation models."
            return (self.create_placeholder_image(), error_msg)

        try:
            # Process reference images (up to 5)
            encoded_images = self.prepare_images_for_api(
                reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5
            )
            has_references = len(encoded_images) > 0
            
            # Build optimized prompt
            final_prompt = self.build_prompt_for_operation(
                prompt, operation, has_references, aspect_ratio, character_consistency
            )
            
            if "Error:" in final_prompt:
                return (self.create_placeholder_image(), final_prompt)
            
            # Add quality instructions
            if quality == "high":
                final_prompt += " Use the highest quality settings available."
            
            # Log operation start
            operation_log = f"NANO BANANA OPERATION LOG\n"
            operation_log += f"Operation: {operation.upper()}\n"
            operation_log += f"Reference Images: {len(encoded_images)}\n"
            operation_log += f"Batch Count: {batch_count}\n"
            operation_log += f"Temperature: {temperature}\n"
            operation_log += f"Quality: {quality}\n"
            operation_log += f"Aspect Ratio: {aspect_ratio}\n"
            operation_log += f"Character Consistency: {character_consistency}\n"
            operation_log += f"Safety Filters: {enable_safety}\n"
            operation_log += f"Note: Output resolution determined by API (max ~1024px)\n"
            operation_log += f"Prompt: {final_prompt[:150]}...\n\n"
            
            # Make API call
            generated_images, api_log = self.call_nano_banana_api(
                final_prompt, encoded_images, temperature, batch_count, enable_safety
            )
            
            operation_log += api_log
            
            # Process results
            if generated_images:
                # Combine all generated images into a batch tensor
                combined_tensor = torch.cat(generated_images, dim=0)
                
                # Calculate approximate cost
                approx_cost = len(generated_images) * 0.039  # ~$0.039 per image
                operation_log += f"\nEstimated cost: ~${approx_cost:.3f}\n"
                operation_log += f"Successfully generated {len(generated_images)} image(s)!"
                
                return (combined_tensor, operation_log)
            else:
                operation_log += "\nNo images were generated. Check the log above for details."
                return (self.create_placeholder_image(), operation_log)
                
        except Exception as e:
            error_log = f"NANO BANANA ERROR: {str(e)}\n"
            error_log += "Please check your API key, internet connection, and paid tier status."
            return (self.create_placeholder_image(), error_log)

# Node registration
NODE_CLASS_MAPPINGS = {
    "ComfyUI_NanoBanana": ComfyUI_NanoBanana,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_NanoBanana": "Nano Banana (Gemini 2.5 Flash Image)",
}