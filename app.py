import os
import torch
import gradio as gr
import phantom_wan
from phantom_wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from phantom_wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from generate import generate, _validate_args
import tempfile
from huggingface_hub import hf_hub_download

def setup_model():
    """Load the model into memory to make running multiple predictions efficient"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    ckpt_dir = "models"
    phantom_ckpt = "models"  # Directory path where the index file is located
    
    # Update config to use local tokenizer path
    for config in WAN_CONFIGS.values():
        if hasattr(config, 't5_tokenizer'):
            config.t5_tokenizer = "."
            
    return ckpt_dir, phantom_ckpt

def generate_video(
    task,
    prompt,
    size,
    frame_num,
    sample_fps,
    base_seed,
    image,
    ref_images,
    sample_solver,
    sample_steps,
    sample_shift,
    sample_guide_scale,
    sample_guide_scale_img,
    sample_guide_scale_text
):
    # Validate inputs
    if task not in WAN_CONFIGS:
        raise gr.Error(f"Unsupported task: {task}")
    if size not in SUPPORTED_SIZES[task]:
        raise gr.Error(f"Unsupported size {size} for task {task}")
    
    # Set default frame_num based on task
    if frame_num is None:
        frame_num = 1 if "t2i" in task else 81
        
    # Validate frame_num for t2i task
    if "t2i" in task and frame_num != 1:
        raise gr.Error(f"frame_num must be 1 for t2i task")
        
    # Prepare ref_image string if provided
    ref_image_str = ",".join([str(p) for p in ref_images]) if ref_images else None
    
    # Create a temporary directory for the output
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output.mp4")
        
    # Prepare arguments
    args = type('Args', (), {
        'task': task,
        'size': size,
        'frame_num': frame_num,
        'sample_fps': sample_fps,
        'ckpt_dir': ckpt_dir,
        'phantom_ckpt': phantom_ckpt,
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
    
    return output_path

# Setup model
ckpt_dir, phantom_ckpt = setup_model()

# Create Gradio interface
with gr.Blocks(title="Phantom Video Generation") as demo:
    gr.Markdown("# Phantom Video Generation")
    
    with gr.Row():
        with gr.Column():
            task = gr.Dropdown(
                choices=["t2v-14B", "t2i-14B", "i2v-14B", "s2v-14B"],
                value="t2v-14B",
                label="Task"
            )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here..."
            )
            size = gr.Dropdown(
                choices=["1280*720", "832*480", "480*832", "480*480"],
                value="1280*720",
                label="Output Size"
            )
            frame_num = gr.Slider(
                minimum=1,
                maximum=121,
                value=81,
                step=4,
                label="Number of Frames (4n+1)"
            )
            sample_fps = gr.Slider(
                minimum=1,
                maximum=60,
                value=24,
                step=1,
                label="FPS"
            )
            base_seed = gr.Number(
                value=-1,
                label="Random Seed (-1 for random)"
            )
            
        with gr.Column():
            image = gr.Image(
                label="Input Image (for i2v task)",
                type="filepath"
            )
            ref_images = gr.File(
                label="Reference Images (for s2v task)",
                file_count="multiple"
            )
            
        with gr.Column():
            sample_solver = gr.Dropdown(
                choices=["unipc", "dpm++"],
                value="unipc",
                label="Sample Solver"
            )
            sample_steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=25,
                step=1,
                label="Sample Steps"
            )
            sample_shift = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=5.0,
                step=0.1,
                label="Sample Shift"
            )
            sample_guide_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=5.0,
                step=0.1,
                label="Guide Scale"
            )
            sample_guide_scale_img = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=5.0,
                step=0.1,
                label="Image Guide Scale"
            )
            sample_guide_scale_text = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.1,
                label="Text Guide Scale"
            )
    
    generate_btn = gr.Button("Generate Video")
    output_video = gr.Video(label="Generated Video")
    
    generate_btn.click(
        fn=generate_video,
        inputs=[
            task,
            prompt,
            size,
            frame_num,
            sample_fps,
            base_seed,
            image,
            ref_images,
            sample_solver,
            sample_steps,
            sample_shift,
            sample_guide_scale,
            sample_guide_scale_img,
            sample_guide_scale_text
        ],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch(share=True) 