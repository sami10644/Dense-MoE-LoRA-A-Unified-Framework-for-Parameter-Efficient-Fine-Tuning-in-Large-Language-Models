# Dense MoE LoRA

### A Unified Framework for Efficient Fine-Tuning of Large Language Models

**Dense MoE LoRA** combines **Mixture-of-Experts (MoE)** with **Low-Rank Adaptation (LoRA)** to fine-tune Large Language Models (LLMs) efficiently. It reduces computational costs while maintaining high accuracy.

## Features
- **Efficient Fine-Tuning**: Uses LoRA with MoE for better resource utilization.
- **Dense Gating Mechanism**: Selects LoRA experts dynamically.
- **Load-Balancing Loss**: Ensures stable training and balanced token distribution.
- **Improved Performance**: Achieves **93% F1 on SST-2** and **84.03% F1 on MRPC**, outperforming standard LoRA.
