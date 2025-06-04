import os
import torch
import cog
from cog import BasePredictor, Path, Input
from typing import List
from enum import Enum
import phantom_wan
from phantom_wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from phantom_wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from generate import generate, _validate_args

class TaskType(str, Enum):
    T2V_1_3B = "t2v-1.3B"
    T2V_14B = "t2v-14B"
    T2I_14B = "t2i-14B"
    I2V_14B = "i2v-14B"
    S2V_1_3B = "s2v-1.3B"
    S2V_14B = "s2v-14B"

class SizeType(str, Enum):
    SIZE_1280_720 = "1280*720"
    SIZE_832_480 = "832*480"
    SIZE_480_832 = "480*832"
    SIZE_480_480 = "480*480"

class SampleSolverType(str, Enum):
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model paths are relative to the project root
        self.ckpt_dir = "models"
        self.phantom_ckpt = "Phantom-Wan-1.3B.pth"

    def predict(
        self,
        task: TaskType = Input(description="The task to run", default=TaskType.T2V_1_3B),
        prompt: str = Input(description="The prompt to generate from"),
        size: SizeType = Input(description="The size of the output (width*height)", default=SizeType.SIZE_832_480),
        frame_num: int = Input(description="Number of frames to generate (4n+1)", default=81, ge=1, le=121),
        sample_fps: int = Input(description="FPS of the generated video", default=24, ge=1, le=60),
        base_seed: int = Input(description="Random seed for generation", default=-1),
        image: Path = Input(description="Input image for i2v task", default=None),
        ref_image: List[Path] = Input(description="Reference image(s) for subject-to-video tasks. For multiple images, provide this input multiple times.", default=None),
        sample_solver: SampleSolverType = Input(description="The solver used to sample", default=SampleSolverType.UNIPC),
        sample_steps: int = Input(description="The sampling steps", default=25, ge=1, le=50),
        sample_guide_scale: float = Input(description="Classifier free guidance scale", default=5.0, ge=1.0, le=20.0),
        sample_guide_scale_img: float = Input(description="Classifier free guidance scale for reference images", default=5.0, ge=1.0, le=20.0),
        sample_guide_scale_text: float = Input(description="Classifier free guidance scale for text", default=7.5, ge=1.0, le=20.0)
    ) -> Path:
        """Run a single prediction on the model"""
        # Validate inputs
        if task.value not in WAN_CONFIGS:
            raise ValueError(f"Unsupported task: {task}")
        if size.value not in SUPPORTED_SIZES[task.value]:
            raise ValueError(f"Unsupported size {size} for task {task}")
        
        # Set default frame_num based on task
        if frame_num is None:
            frame_num = 1 if "t2i" in task.value else 81
            
        # Validate frame_num for t2i task
        if "t2i" in task.value and frame_num != 1:
            raise ValueError(f"frame_num must be 1 for t2i task")
            
        # Prepare ref_image string if provided
        ref_image_str = ",".join([str(p) for p in ref_image]) if ref_image else None
            
        # Prepare arguments
        args = type('Args', (), {
            'task': task.value,
            'size': size.value,
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
            'sample_solver': sample_solver.value,
            'sample_steps': sample_steps,
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