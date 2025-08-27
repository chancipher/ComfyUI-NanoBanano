"""
ComfyUI Nano Banana Node
A ComfyUI custom node for Google's Gemini 2.5 Flash Image (Nano Banana) model.

Author: Your Name
Version: 1.0.0
License: MIT
"""

from .nano_banano import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Expose the mappings for ComfyUI
WEB_DIRECTORY = "./web"