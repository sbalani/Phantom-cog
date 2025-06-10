import os
import torch
import cog
from cog import BasePredictor, Path, Input
from typing import List
import phantom_wan
from phantom_wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from phantom_wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from generate import generate, _validate_args
import tempfile
from huggingface_hub import hf_hub_download

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Download model files if they don't exist, using huggingface_hub
        model_files = {
            "models_t5_umt5-xxl-enc-bf16.pth": {
                "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "filename": "models_t5_umt5-xxl-enc-bf16.pth"
            },
            "Wan2.1_VAE.pth": {
                "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "filename": "Wan2.1_VAE.pth"
            },
            "diffusion_pytorch_model.safetensors": {
                "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "filename": "diffusion_pytorch_model.safetensors"
            },
            "Phantom_Wan_14B.safetensors.index.json": {
                "repo_id": "bytedance-research/Phantom",
                "filename": "Phantom_Wan_14B.safetensors.index.json"
            },
            # Tokenizer files
            "config.json": {
                "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "filename": "config.json"
            },
            "special_tokens_map.json": {
                "repo_id": "google/umt5-xxl",
                "filename": "special_tokens_map.json"
            },
            "tokenizer_config.json": {
                "repo_id": "google/umt5-xxl",
                "filename": "tokenizer_config.json"
            },
            "spiece.model": {
                "repo_id": "google/umt5-xxl",
                "filename": "spiece.model"
            }
        }
        
        for filename, info in model_files.items():
            filepath = os.path.join("models", filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename} from {info['repo_id']}...")
                hf_hub_download(
                    repo_id=info["repo_id"],
                    filename=info["filename"],
                    local_dir="models",
                    local_dir_use_symlinks=False
                )
        
        # Download 14B model shards using huggingface_hub
        model_shards = {
            "model-00001-of-00006.safetensors": "Phantom_Wan_14B-00001-of-00006.safetensors",
            "model-00002-of-00006.safetensors": "Phantom_Wan_14B-00002-of-00006.safetensors",
            "model-00003-of-00006.safetensors": "Phantom_Wan_14B-00003-of-00006.safetensors",
            "model-00004-of-00006.safetensors": "Phantom_Wan_14B-00004-of-00006.safetensors",
            "model-00005-of-00006.safetensors": "Phantom_Wan_14B-00005-of-00006.safetensors",
            "model-00006-of-00006.safetensors": "Phantom_Wan_14B-00006-of-00006.safetensors",
            "model.safetensors.index.json": "Phantom_Wan_14B.safetensors.index.json"
        }
        
        for filename, repo_filename in model_shards.items():
            filepath = os.path.join("models", filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename} from bytedance-research/Phantom...")
                hf_hub_download(
                    repo_id="bytedance-research/Phantom",
                    filename=repo_filename,
                    local_dir="models",
                    local_dir_use_symlinks=False
                )
        
        # Model paths are relative to the project root
        self.ckpt_dir = "models"
        self.phantom_ckpt = "models"  # Directory path where the index file is located
        
        # Update config to use local tokenizer path
        for config in WAN_CONFIGS.values():
            if hasattr(config, 't5_tokenizer'):
                config.t5_tokenizer = "."

    def predict(
        self,
        task: str = Input(
            description="The task to run",
            default="t2v-14B",
            choices=["t2v-14B", "t2i-14B", "i2v-14B", "s2v-14B"]
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
        
        # Create a temporary directory for the output
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")
            
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
            'sample_guide_scale_text': sample_guide_scale_text,
            'save_file': output_path
        })()
        
        # Validate arguments
        _validate_args(args)
        
        # Generate output
        generate(args)
        
        # Return the output path
        return Path(output_path)