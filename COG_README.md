# Phantom Model Cog Deployment

This repository contains the Cog configuration for deploying the Phantom model to Replicate.

## Prerequisites

1. Install Cog:
```bash
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
chmod +x /usr/local/bin/cog
```

2. Install Docker if not already installed:
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io
```

## Model Setup

1. Download the model checkpoints from [bytedance-research/Phantom](https://github.com/bytedance-research/Phantom)

2. Set up the environment variables for the model paths:
```bash
export PHANTOM_CKPT_DIR=/path/to/checkpoint/directory
export PHANTOM_CKPT=/path/to/phantom/checkpoint
```

## Building and Testing

1. Build the Cog model:
```bash
cog build
```

2. Test the model locally:
```bash
cog predict -i task="t2v-1.3B" -i prompt="Your prompt here"
```

## Deploying to Replicate

1. Create a Replicate account and get your API token

2. Log in to Replicate:
```bash
cog login
```

3. Push the model to Replicate:
```bash
cog push
```

## Model Parameters

The model accepts the following parameters:

- `task`: The task to run (t2v-1.3B, t2v-14B, t2i-14B, i2v-14B)
- `prompt`: The text prompt to generate from
- `size`: Output size in format "width*height" (default: "1280*720")
- `frame_num`: Number of frames to generate (default: 81 for video, 1 for image)
- `sample_fps`: FPS of the generated video
- `base_seed`: Random seed for generation
- `image`: Input image for i2v task

## Example Usage

```python
import replicate

# Text to Video
output = replicate.run(
    "your-username/phantom",
    input={
        "task": "t2v-1.3B",
        "prompt": "Two cats fighting in a boxing ring",
        "size": "1280*720"
    }
)

# Image to Video
output = replicate.run(
    "your-username/phantom",
    input={
        "task": "i2v-14B",
        "prompt": "Transform this into a summer beach scene",
        "image": open("input.jpg", "rb")
    }
)
``` 