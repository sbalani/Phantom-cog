import os
import torch
import cog
from cog import BasePredictor, Path, Input
from typing import List
import phantom_wan
from phantom_wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from phantom_wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from generate import generate, _validate_args
import subprocess

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Download model files if they don't exist
        model_files = {
            "models_t5_umt5-xxl-enc-bf16.pth": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth?download=true",
            "Phantom-Wan-1.3B.pth": "https://huggingface.co/bytedance-research/Phantom/resolve/main/Phantom-Wan-1.3B.pth?download=true",
            # Tokenizer files
            "tokenizer/config.json": "https://huggingface.co/google/umt5-xxl/resolve/main/config.json",
            "tokenizer/special_tokens_map.json": "https://huggingface.co/google/umt5-xxl/resolve/main/special_tokens_map.json",
            "tokenizer/tokenizer_config.json": "https://huggingface.co/google/umt5-xxl/resolve/main/tokenizer_config.json",
            "tokenizer/spiece.model": "https://huggingface.co/google/umt5-xxl/resolve/main/spiece.model"
        }
        
        for filename, url in model_files.items():
            filepath = os.path.join("models", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                subprocess.run(["wget", "-O", filepath, url], check=True)
        
        # Model paths are relative to the project root
        self.ckpt_dir = "models"
        self.phantom_ckpt = "Phantom-Wan-1.3B.pth"
        
        # Update config to use local tokenizer path
        for config in WAN_CONFIGS.values():
            if hasattr(config, 't5_tokenizer'):
                config.t5_tokenizer = "tokenizer"

    def predict(
        self,
        task: str = Input(
            description="The task to run",
            default="t2v-1.3B",
            choices=["t2v-1.3B", "t2v-14B", "t2i-14B", "i2v-14B", "s2v-1.3B", "s2v-14B"]
        ),
        prompt: str = Input(description="The prompt to generate from"),
        size: str = Input(
            description="The size of the output (width*height)",
            default="1280*720",
            choices=["1280*720", "832*480", "480*832", "480*480"]
        ),
        frame_num: int = Input(
            description="Number of frames to generate (4n+1)",
            default=81,
            ge=1,
            le=121
        ),
        sample_fps: int = Input(
            description="FPS of the generated video",
            default=24,
            ge=1,
            le=60
        ),
        base_seed: int = Input(
            description="Random seed for generation",
            default=-1
        ),
        image: Path = Input(
            description="Input image for i2v task",
            default=None
        ),
        ref_image: List[Path] = Input(
            description="Reference image(s) for subject-to-video tasks. For multiple images, provide this input multiple times.",
            default=None
        ),
        sample_solver: str = Input(
            description="The solver used to sample",
            default="unipc",
            choices=["unipc", "dpm++"]
        ),
        sample_steps: int = Input(
            description="The sampling steps",
            default=25,
            ge=1,
            le=50
        ),
        sample_shift: float = Input(
            description="Noise schedule shift parameter. Affects temporal dynamics. For 480p videos, it's recommended to use 3.0",
            default=5.0,
            ge=1.0,
            le=10.0
        ),
        sample_guide_scale: float = Input(
            description="Classifier free guidance scale",
            default=5.0,
            ge=1.0,
            le=20.0
        ),
        sample_guide_scale_img: float = Input(
            description="Classifier free guidance scale for reference images",
            default=5.0,
            ge=1.0,
            le=20.0
        ),
        sample_guide_scale_text: float = Input(
            description="Classifier free guidance scale for text",
            default=7.5,
            ge=1.0,
            le=20.0
        )
    ) -> Path:
        """Run a single prediction on the model"""
        # Validate inputs
        if task not in WAN_CONFIGS:
            raise ValueError(f"Unsupported task: {task}")
        if size not in SUPPORTED_SIZES[task]:
            raise ValueError(f"Unsupported size {size} for task {task}")
        
        # Set default frame_num based on task
        if frame_num is None:
            frame_num = 1 if "t2i" in task else 81
            
        # Validate frame_num for t2i task
        if "t2i" in task and frame_num != 1:
            raise ValueError(f"frame_num must be 1 for t2i task")
            
        # Prepare ref_image string if provided
        ref_image_str = ",".join([str(p) for p in ref_image]) if ref_image else None
            
        # Prepare arguments
        args = type('Args', (), {
            'task': task,
            'size': size,
            'frame_num': frame_num,
            'sample_fps': sample_fps,
            'ckpt_dir': self.ckpt_dir,
            'phantom_ckpt': self.phantom_ckpt,
            'prompt': prompt,
            'base_seed': base_seed,
            'image': str(image) if image else None,
            'ref_image': ref_image_str,
            'offload_model': True,
            'ulysses_size': 1,
            'ring_size': 1,
            't5_fsdp': False,
            't5_cpu': False,
            'dit_fsdp': False,
            'use_prompt_extend': False,
            'sample_solver': sample_solver,
            'sample_steps': sample_steps,
            'sample_shift': sample_shift,
            'sample_guide_scale': sample_guide_scale,
            'sample_guide_scale_img': sample_guide_scale_img,
            'sample_guide_scale_text': sample_guide_scale_text
        })()
        
        # Validate arguments
        _validate_args(args)
        
        # Generate output
        output = generate(args)
        
        # Return the output path
        return Path(output) 