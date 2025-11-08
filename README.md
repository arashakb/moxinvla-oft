# MoxinVLA-OFT: Vision-Language-Action Model with Moxin LLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**MoxinVLA-OFT** is a fine-tuning framework for Vision-Language-Action (VLA) models that uses the **Moxin** large language model as the language backbone, built upon the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) framework.

---

## 📋 Table of Contents

- [What is MoxinVLA?](#-what-is-moxinvla)
- [What is Moxin?](#-what-is-moxin)
- [Key Features](#-key-features)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
  - [General Setup](#general-setup)
  - [LIBERO Setup](#libero-setup)
- [Quick Start](#-quick-start)
- [Training & Evaluation](#-training--evaluation)
- [News & Updates](#-news--updates)
- [Project Structure](#-project-structure)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)
- [Support](#-support)

---

## 🤖 What is MoxinVLA?

**MoxinVLA-OFT** adapts the OpenVLA architecture to use **Moxin**, an open-source large language model, as the language backbone instead of the original LLaMA-2/Vicuna models. This enables:

- Training VLA models with an alternative LLM backbone
- Exploring the impact of different language models on robotic manipulation
- Fine-tuning on custom robot datasets using the OFT (Optimized Fine-Tuning) methodology

The model architecture follows the OpenVLA design: a vision encoder (DINOv2 + SigLIP) combined with a language model (Moxin) to output continuous robot actions.

---

## 🧠 What is Moxin?

**Moxin** is an open-source 7B parameter large language model based on the Mistral architecture. It is developed by [moxin-org](https://huggingface.co/moxin-org) and serves as the language backbone in MoxinVLA.

**Key characteristics:**
- 7 billion parameters
- Based on Mistral architecture
- Compatible with MistralForCausalLM
- Available on HuggingFace: [`moxin-org/moxin-llm-7b`](https://huggingface.co/moxin-org/moxin-llm-7b)

---

## ✨ Key Features

- **Moxin LLM Integration**: Uses Moxin as the language backbone
- **OFT Fine-tuning**: Efficient fine-tuning with LoRA (rank-32)
- **Multi-camera Support**: Handles 1-3 camera inputs (third-person + wrist cameras)
- **Proprioceptive State**: Includes robot joint state in the model input
- **LIBERO Support**: Fine-tune and evaluate on LIBERO benchmark tasks
- **ALOHA Support**: Fine-tune on real-world ALOHA robot data
- **Simple Logging**: Clean logging system for training and evaluation
- **WandB Integration**: Optional Weights & Biases logging

---

## 💻 System Requirements

### Inference
- 1 GPU with ~16 GB VRAM for LIBERO simulation tasks
- 1 GPU with ~18 GB VRAM for ALOHA robot tasks

### Training
- **Minimum**: 1 GPU with 27+ GB VRAM (e.g., RTX 3090, A6000)
- **Recommended**: 4-8 GPUs with 40-80 GB VRAM (e.g., A100, H100)
- Training time varies: ~6-12 hours for 50K steps on 8xH100

**Note**: Training performance may vary across different GPU types. It's recommended to test and evaluate on the same GPU used for training.

---

## 🚀 Installation

### General Setup

1. **Create conda environment**
```bash
conda create -n moxinvla-oft python=3.10 -y
conda activate moxinvla-oft
```

2. **Install PyTorch**
```bash
# For CUDA 11.8 (adjust based on your system)
pip3 install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

3. **Clone and install MoxinVLA-OFT**
```bash
git clone https://github.com/arashakb/moxinvla-oft.git
cd moxinvla-oft
pip install -e .
```

4. **Install Flash Attention 2** (required for training)
```bash
pip install packaging ninja
ninja --version  # Should return 0
pip install "flash-attn==2.5.5" --no-build-isolation
```

> **Troubleshooting**: If Flash Attention installation fails, try `pip cache remove flash_attn` first.

### LIBERO Setup

To train or evaluate on LIBERO benchmark:

1. **Install LIBERO**
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
```

2. **Install LIBERO requirements**
```bash
pip install -r experiments/robot/libero/libero_requirements.txt
# update the following packages to these specific versions to avoid conflicts
pip install numpy==1.26.4
pip install opencv-python==4.11.0.86
```

3. **(Optional) Download LIBERO datasets**

If you plan to fine-tune, download the preprocessed LIBERO datasets (~10 GB):
```bash
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
```

This includes:
- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-10 (Long)

> **Note**: Pretrained checkpoints are provided, so dataset download is optional if you only want to evaluate.

---

## 🎯 Quick Start

### Basic Inference Example

```python
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Configure the model
cfg = GenerateConfig(
    pretrained_checkpoint="path/to/your/checkpoint",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    center_crop=True,
    unnorm_key="libero_spatial_no_noops",
)

# Load model components
vla = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load observation
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as f:
    observation = pickle.load(f)

# Generate actions
actions = get_vla_action(
    cfg, vla, processor, observation, 
    observation["task_description"], 
    action_head, proprio_projector
)

print("Generated action chunk:", actions)
```

---

## 📚 Training & Evaluation

### Fine-tuning on LIBERO

**Using the training script:**
```bash
bash libero_finetune.sh
```

**Manual training command:**
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path path/to/moxin-checkpoint \
  --data_root_dir modified_libero_rlds_data \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir logs/checkpoints/my-experiment \
  --lora_rank 32 \
  --batch_size 8 \
  --learning_rate 3e-4 \
  --use_film False \
  --use_proprio True \
  --num_images_in_input 2 \
  --max_steps 50000
```

**Configuration:**
- Adjust `--nproc-per-node` based on available GPUs
- Modify `--batch_size` based on GPU memory
- Set `WANDB_MODE=disabled` to disable WandB logging

See [LIBERO.md](LIBERO.md) for detailed LIBERO training/evaluation instructions.

See [ALOHA.md](ALOHA.md) for ALOHA robot setup and usage.

### Evaluation

```bash
bash libero_evaluation.sh
```

**Monitor logs:**
```bash
# Watch training progress
tail -f logs/training/*.log

# Watch evaluation progress
tail -f logs/evaluation/*.log
```

---

## 📰 News & Updates

**🚀 Coming Soon**: We will be releasing pretrained MoxinVLA checkpoints fine-tuned on LIBERO benchmarks for the community to use and build upon.

**Current Status**: 
- Base MoxinVLA model integration is complete
- Fine-tuning and evaluation scripts are ready
- Training on LIBERO benchmarks in progress

Stay tuned for checkpoint releases on HuggingFace!

---

## 📁 Project Structure

```
moxinvla-oft/
├── prismatic/                  # Core VLA model code
│   ├── models/
│   │   └── backbones/llm/
│   │       └── moxin.py       # Moxin LLM integration
│   └── vla/                   # VLA-specific code
├── vla-scripts/
│   ├── finetune.py            # Fine-tuning script
│   └── deploy.py              # Deployment script
├── experiments/robot/
│   ├── libero/                # LIBERO evaluation
│   └── aloha/                 # ALOHA evaluation
├── scripts/
│   └── logging_utils.sh       # Logging utilities
├── libero_finetune.sh         # LIBERO training script
├── libero_evaluation.sh       # LIBERO evaluation script
├── LIBERO.md                  # LIBERO documentation
├── ALOHA.md                   # ALOHA documentation
└── SETUP.md                   # Detailed setup instructions
```

---

<!--
## 📖 Citation

If you use MoxinVLA-OFT in your research, please cite both this work and the original OpenVLA-OFT paper:

```bibtex
@misc{akbarinia2025moxinvla,
  title={MoxinVLA-OFT: Fine-Tuning Vision-Language-Action Models with Moxin LLM},
  author={Akbarinia, Arash},
  year={2025},
  url={https://github.com/arashakb/moxinvla-oft}
}

@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
-->

---

## 🙏 Acknowledgements

This project builds upon the excellent work of:

- **[OpenVLA-OFT](https://github.com/moojink/openvla-oft)** by Moo Jin Kim, Chelsea Finn, and Percy Liang
- **[OpenVLA](https://github.com/openvla/openvla)** - The original OpenVLA project
- **[Moxin](https://huggingface.co/moxin-org)** - The Moxin language model
- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** - Lifelong robot learning benchmark

We are grateful to the original authors for making their code publicly available.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Original OpenVLA-OFT code is also under MIT License. Moxin model weights follow their respective license terms.

---

## 💬 Support

For questions or issues, please open a [GitHub Issue](https://github.com/arashakb/moxinvla-oft/issues).

---

