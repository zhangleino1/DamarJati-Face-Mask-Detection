# DamarJati-Face-Mask-Detection



This model is a fine-tuned version of microsoft/swin-tiny-patch4-window7-224 on the imagefolder dataset. It achieves the following results on the evaluation set:

Loss: 0.0051
Accuracy: 0.9992
Model description
More information needed

Intended uses & limitations
More information needed

Training and evaluation data
More information needed

Training procedure
Training hyperparameters
The following hyperparameters were used during training:

learning_rate: 5e-05
train_batch_size: 32
eval_batch_size: 32
seed: 42
gradient_accumulation_steps: 4
total_train_batch_size: 128
optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
lr_scheduler_type: linear
lr_scheduler_warmup_ratio: 0.1
num_epochs: 3
Training results
Training Loss	Epoch	Step	Validation Loss	Accuracy
0.0344	1.0	83	0.0051	0.9992
0.0112	2.0	166	0.0052	0.9983
0.0146	3.0	249	0.0045	0.9992
Framework versions
Transformers 4.34.0
Pytorch 2.0.1+cu118
Datasets 2.14.5
Tokenizers 0.14.0


# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("Heem2/Facemask-detection")
model = AutoModelForImageClassification.from_pretrained("Heem2/Facemask-detection")