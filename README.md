# layoutlmv3-receipt-understanding
Fine-tuning pre-trained LayoutLMv3 model for receipt information understanding

## Load Dataset

Load the dataset using the `datasets` library from Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("hcsun/cord")
```

## Load Fine-Tuned Model

Load the fine-tuned model directly using the transformers library from Hugging Face:

```python
from transformers import AutoProcessor, AutoModelForTokenClassification

processor = AutoProcessor.from_pretrained("hcsun/layoutlmv3-cord")
model = AutoModelForTokenClassification.from_pretrained("hcsun/layoutlmv3-cord")
```