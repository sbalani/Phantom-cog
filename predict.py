import os
import torch
import cog
from pathlib import Path
import phantom_wan
from phantom_wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from phantom_wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

class Predictor(cog.Predictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model paths will be set during cog push
        self.ckpt_dir = os.environ.get("PHANTOM_CKPT_DIR")
        self.phantom_ckpt = os.environ.get("PHANTOM_CKPT")
        
        if not self.ckpt_dir or not self.phantom_ckpt:
            raise ValueError("Please set PHANTOM_CKPT_DIR and PHANTOM_CKPT environment variables")

    @cog.input("task", type=str, default="t2v-1.3B", 
               help="The task to run (t2v-1.3B, t2v-14B, t2i-14B, i2v-14B)")
    @cog.input("prompt", type=str, help="The prompt to generate from")
    @cog.input("size", type=str, default="1280*720", 
               help="The size of the output (width*height)")
    @cog.input("frame_num", type=int, default=None,
               help="Number of frames to generate (4n+1)")
    @cog.input("sample_fps", type=int, default=None,
               help="FPS of the generated video")
    @cog.input("base_seed", type=int, default=-1,
               help="Random seed for generation")
    @cog.input("image", type=cog.Path, default=None,
               help="Input image for i2v task")
    def predict(self, task, prompt, size="1280*720", frame_num=None, 
                sample_fps=None, base_seed=-1, image=None):
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
            'offload_model': True,
            'ulysses_size': 1,
            'ring_size': 1,
            't5_fsdp': False,
            't5_cpu': False,
            'dit_fsdp': False,
            'use_prompt_extend': False,
        })()
        
        # Generate output
        output = phantom_wan.generate(args)
        
        # Return the output path
        return Path(output) 