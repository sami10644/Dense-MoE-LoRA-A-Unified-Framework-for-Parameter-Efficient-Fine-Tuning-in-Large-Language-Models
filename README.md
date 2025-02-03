Dense MoE LoRA: A Unified Framework for Parameter-Efficient Fine-Tuning in Large Language Models



Overview
Dense MoE LoRA is a novel framework that combines Mixture-of-Experts (MoE) and Low-Rank Adaptation (LoRA) to achieve parameter-efficient fine-tuning (PEFT) of Large Language Models (LLMs). This approach enables better resource utilization by incorporating dense gating for LoRA expert selection, along with an auxiliary load-balancing loss to ensure stable training and equitable token distribution across experts.

Key Features
✅ Integrates LoRA with Dense MoE for fine-tuning large models efficiently.
✅ Uses dense gating for expert selection within the feed-forward layers of a Transformer.
✅ Includes auxiliary load-balancing loss for stable training.
✅ Achieves 93% F1 Score on SST-2 and 84.03% F1 Score on MRPC, outperforming baseline LoRA.

Table of Contents
Installation
Usage
Dataset
Training
Evaluation
Results
Contributors
License
Installation
Requirements
Ensure you have the following dependencies installed:

bash
Copy
Edit
pip install torch transformers datasets accelerate bitsandbytes
If using GPU, ensure you have CUDA installed for PyTorch acceleration.

Usage
1. Load Pretrained Model with Dense MoE LoRA
python
Copy
Edit
from model import DenseMoELoRA
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
moe_lora_model = DenseMoELoRA(base_model, num_experts=4, lora_rank=8)
2. Fine-Tune on SST-2 Dataset
python
Copy
Edit
from training import train_model
train_model(moe_lora_model, dataset="sst2", epochs=3, batch_size=16, lr=2e-5)
3. Evaluate on MRPC Dataset
python
Copy
Edit
from evaluation import evaluate_model
evaluate_model(moe_lora_model, dataset="mrpc")
Dataset
The model supports fine-tuning on multiple NLP datasets using the Hugging Face Datasets library. By default, the following datasets are used:

SST-2 (Stanford Sentiment Treebank v2) for sentiment classification
MRPC (Microsoft Research Paraphrase Corpus) for paraphrase detection
You can easily modify the dataset in training.py.

Training
To train the model from scratch on a dataset:

bash
Copy
Edit
python train.py --dataset sst2 --epochs 3 --batch_size 16 --lr 2e-5
To fine-tune an already trained model:

bash
Copy
Edit
python train.py --dataset mrpc --load_checkpoint "checkpoints/moe_lora_sst2.pt"
Evaluation
Evaluate the model’s performance on SST-2 or MRPC:

bash
Copy
Edit
python evaluate.py --dataset mrpc --load_checkpoint "checkpoints/moe_lora_sst2.pt"
Results
Model	SST-2 F1 Score	MRPC F1 Score
LoRA	92.77%	81.78%
MoE LoRA	93.00%	84.03%
Contributors
Abdullah As Sami - University of South Florida
Nafis Saami Azad - University of South Florida
License
This project is licensed under the MIT License. See the LICENSE file for details.

🚀 Star this repo if you find it useful! 🌟
