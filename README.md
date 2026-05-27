# Qwen Radiology Report Generation & Evaluation Toolkit

A toolkit for generating radiology reports from chest X-ray images using Qwen Vision-Language Models, and evaluating the generated reports with multiple clinical and NLG metrics.

> 📊 **[Jump to Experimental Results ↓](#-experimental-results)**

## 📁 Project Structure

```
.
├── qwen_report_generation.py      # Qwen VL inference for report generation
├── evaluate_chexbert.py           # CheXbert-based clinical accuracy evaluation
├── evaluate_nlg.py                # NLG metrics (BLEU, ROUGE, METEOR, BERTScore, etc.)
├── evaluate_llm_as_labeler.py     # LLM-as-labeler clinical accuracy evaluation
├── CheXbert/                      # CheXbert label extraction module
│   └── src/
│       ├── label.py
│       ├── constants.py
│       ├── utils.py
│       ├── bert_tokenizer.py
│       ├── models/
│       │   └── bert_labeler.py
│       └── datasets_chexbert/
│           └── unlabeled_dataset.py
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

For optional clinical metrics:
```bash
# RadGraph F1 (requires model download)
pip install radgraph

# RaTEScore
pip install ratescore
```

## 📋 Prerequisites

### Data Format

**Input dataset** (`test_dataset.json`):
```json
{
  "test": [
    {
      "id": "sample_001",
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/chest_xray.jpg"},
        {"type": "text", "text": "Please generate a radiology report for this chest X-ray."}
      ]
    }
  ]
}
```

**Annotation file** (`annotation.json`) for evaluation:
```json
{
  "test": [
    {
      "id": "sample_001",
      "report": "No acute cardiopulmonary abnormality. The heart size is normal..."
    }
  ]
}
```

### CheXbert Checkpoint

Download the CheXbert checkpoint from:
- https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9

Place it at `./checkpoints/chexbert.pth` (or specify via `--chexbert_checkpoint`).

## 📖 Usage

### 1. Report Generation

Generate radiology reports from chest X-ray images using Qwen VL models:

```bash
# Using Qwen2.5-VL-7B (default)
CUDA_VISIBLE_DEVICES=0,1 python qwen_report_generation.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --question_file ./data/test_dataset.json \
    --output_file ./results/qwen_output.json \
    --max_tokens 512

# Using Qwen3-VL-8B
CUDA_VISIBLE_DEVICES=0,1 python qwen_report_generation.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --model_type qwen3vl \
    --question_file ./data/test_dataset.json \
    --output_file ./results/qwen3vl_output.json \
    --max_tokens 512

# With thinking mode (Qwen3.5)
CUDA_VISIBLE_DEVICES=0,1,2,3 python qwen_report_generation.py \
    --model_name Qwen/Qwen3.5-27B \
    --question_file ./data/test_dataset.json \
    --output_file ./results/qwen35_output.json \
    --max_tokens 4096 \
    --enable_thinking

# Resume from interrupted run
python qwen_report_generation.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --question_file ./data/test_dataset.json \
    --output_file ./results/qwen_output.json \
    --resume
```

**Key arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | HuggingFace model name or local path | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `--model_type` | `auto` or `qwen3vl` | `auto` |
| `--max_tokens` | Max new tokens to generate | `2048` |
| `--enable_thinking` | Enable thinking mode (Qwen3.5) | `False` |
| `--flash_attn` | Use Flash Attention 2 | `False` |
| `--resume` | Resume from existing output | `False` |

### 2. CheXbert Evaluation

Evaluate clinical accuracy using CheXbert label extraction:

```bash
python evaluate_chexbert.py \
    --llm_output ./results/qwen_output.json \
    --annotation_json ./data/annotation.json \
    --chexbert_checkpoint ./checkpoints/chexbert.pth \
    --output_dir ./results/chexbert_eval
```

**Output:**
- `*_eval_summary.xlsx` — Per-disease AUC, F1, Recall, Specificity + macro average
- `*_per_sample.csv` — Per-sample binary predictions and agreements

### 3. NLG Metrics Evaluation

Evaluate with BLEU, ROUGE-L, METEOR, BERTScore, RadGraph F1, RaTEScore:

```bash
# Basic NLG metrics (fast, no GPU needed for BLEU/ROUGE/METEOR)
python evaluate_nlg.py \
    --llm_output ./results/qwen_output.json \
    --annotation_json ./data/annotation.json \
    --output_dir ./results/nlg_eval \
    --metrics bleu,rouge,meteor

# All metrics including BERTScore (needs GPU)
python evaluate_nlg.py \
    --llm_output ./results/qwen_output.json \
    --annotation_json ./data/annotation.json \
    --output_dir ./results/nlg_eval \
    --metrics bleu,rouge,meteor,bertscore

# Full evaluation with clinical metrics
python evaluate_nlg.py \
    --llm_output ./results/qwen_output.json \
    --annotation_json ./data/annotation.json \
    --output_dir ./results/nlg_eval \
    --metrics bleu,rouge,meteor,bertscore,radgraph,ratescore
```

**Available metrics:**
| Metric | Package | GPU Required |
|--------|---------|:---:|
| BLEU-1/2/3/4 | nltk | ❌ |
| ROUGE-L | rouge-score | ❌ |
| METEOR | nltk | ❌ |
| BERTScore | bert-score | ✅ |
| RadGraph F1 | radgraph | ✅ |
| RaTEScore | ratescore | ✅ |

### 4. LLM-as-Labeler Evaluation

Use another LLM as a label extractor (alternative to CheXbert):

```bash
CUDA_VISIBLE_DEVICES=0,1 python evaluate_llm_as_labeler.py \
    --llm_output ./results/qwen_output.json \
    --annotation_json ./data/annotation.json \
    --labeler_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./results/llm_labeler_eval
```

**Features:**
- Persistent GT cache: ground truth reports are labeled only once per labeler model
- Resume support: interrupted runs can be continued
- Both prediction and GT reports are labeled by the same LLM for fair comparison

## 📊 Output Format

The report generation script outputs:
```json
{
  "test": [
    {
      "id": "sample_001",
      "output": "The heart size is normal. The lungs are clear..."
    },
    {
      "id": "sample_002",
      "output": "There is mild cardiomegaly. Small bilateral pleural effusions..."
    }
  ]
}
```

## 📈 Experimental Results

Below are our evaluation results comparing different models on the MIMIC-CXR test set.

### CheXbert as Labeler

| | Qwen3.5-27B | | | | Qwen3-VL-8B | | | |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Disease** | **AUC** | **F1** | **Recall** | **Spec** | **AUC** | **F1** | **Recall** | **Spec** |
| Enlarged Cardiomediastinum | 0.4955 | 0.1097 | 0.1921 | 0.7988 | 0.4758 | 0.1303 | 0.5424 | 0.4092 |
| Cardiomegaly | 0.6989 | 0.6551 | 0.9193 | 0.4784 | 0.5827 | 0.5703 | 0.8924 | 0.273 |
| Lung Opacity | 0.5988 | 0.5238 | 0.5643 | 0.6332 | 0.5772 | 0.5359 | 0.0694 | 0.485 |
| Lung Lesion | 0.5072 | 0.0482 | 0.0317 | 0.9827 | 0.5001 | 0.0337 | 0.0238 | 0.9765 |
| Edema | 0.6842 | 0.4471 | 0.6485 | 0.7198 | 0.5624 | 0.3115 | 0.4653 | 0.6595 |
| Consolidation | 0.5145 | 0.0671 | 0.0485 | 0.9805 | 0.5141 | 0.0797 | 0.1165 | 0.9117 |
| Pneumonia | 0.5529 | 0.1505 | 0.1400 | 0.9659 | 0.5074 | 0.0354 | 0.02 | 0.9948 |
| Atelectasis | 0.5513 | 0.2854 | 0.2153 | 0.8873 | 0.5008 | 0.0128 | 0.0065 | 0.995 |
| Pneumothorax | 0.6280 | 0.3171 | 0.2653 | 0.9907 | 0.5081 | 0.0339 | 0.0204 | 0.9958 |
| Pleural Effusion | 0.7105 | 0.6190 | 0.6190 | 0.8019 | 0.5505 | 0.2796 | 0.1905 | 0.9106 |
| Pleural Other | 0.5142 | 0.0556 | 0.0294 | 0.9991 | 0.4958 | 0 | 0 | 0.9916 |
| Fracture | 0.4995 | 0.0000 | 0.0000 | 0.9990 | 0.499 | 0 | 0 | 0.9981 |
| Support Devices | 0.6783 | 0.5688 | 0.4547 | 0.9019 | 0.645 | 0.5574 | 0.5128 | 0.7772 |
| No Finding | 0.6520 | 0.2557 | 0.4242 | 0.8797 | 0.5023 | 0.0144 | 0.0076 | 0.9971 |
| **Macro Average** | **0.5918** | **0.2931** | **0.3252** | **0.8585** | **0.5301** | **0.1854** | **0.2477** | **0.8125** |

### Qwen3.5 as Labeler

| | Qwen3.5-27B | | | | Qwen3-VL-8B | | | |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Disease** | **AUC** | **F1** | **Recall** | **Spec** | **AUC** | **F1** | **Recall** | **Spec** |
| Enlarged Cardiomediastinum | 0.6108 | 0.3622 | 0.6525 | 0.5691 | 0.5487 | 0.3088 | 0.5875 | 0.5099 |
| Cardiomegaly | 0.7090 | 0.6466 | 0.9278 | 0.4902 | 0.5925 | 0.5561 | 0.8789 | 0.3061 |
| Lung Opacity | 0.6253 | 0.5931 | 0.6915 | 0.5592 | 0.5938 | 0.5899 | 0.7758 | 0.4118 |
| Lung Lesion | 0.5288 | 0.1085 | 0.0619 | 0.9957 | 0.4950 | 0.0000 | 0.0000 | 0.9900 |
| Edema | 0.7031 | 0.5464 | 0.6577 | 0.7486 | 0.5773 | 0.3986 | 0.5207 | 0.6338 |
| Consolidation | 0.5263 | 0.0990 | 0.0926 | 0.9600 | 0.5049 | 0.0721 | 0.1111 | 0.8987 |
| Pneumonia | 0.5689 | 0.1200 | 0.1800 | 0.9579 | 0.5012 | 0.0225 | 0.0200 | 0.9824 |
| Atelectasis | 0.5547 | 0.3080 | 0.2324 | 0.8771 | 0.5046 | 0.0446 | 0.0235 | 0.9856 |
| Pneumothorax | 0.6533 | 0.3373 | 0.3182 | 0.9885 | 0.4979 | 0 | 0.0000 | 0.9958 |
| Pleural Effusion | 0.7217 | 0.6276 | 0.6293 | 0.8141 | 0.5635 | 0.3197 | 0.2298 | 0.8972 |
| Pleural Other | 0.5123 | 0.0482 | 0.0250 | 0.9995 | 0.5085 | 0.0404 | 0.0250 | 0.9920 |
| Fracture | 0.4998 | 0.0000 | 0.0000 | 0.9995 | 0.5000 | 0.0000 | 0.0000 | 1.0000 |
| Support Devices | 0.7365 | 0.8042 | 0.8640 | 0.6090 | 0.6835 | 0.7376 | 0.7451 | 0.6219 |
| No Finding | 0.8213 | 0.2918 | 0.8571 | 0.7855 | 0.7823 | 0.3162 | 0.7143 | 0.8503 |
| **Macro Average** | **0.6266** | **0.3495** | **0.4421** | **0.8110** | **0.5610** | **0.2433** | **0.3308** | **0.7911** |

### Difference (Qwen3.5 as Labeler − CheXbert as Labeler)

| | Qwen3.5-27B | | | | Qwen3-VL-8B | | | |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Disease** | **ΔAUC** | **ΔF1** | **ΔRecall** | **ΔSpec** | **ΔAUC** | **ΔF1** | **ΔRecall** | **ΔSpec** |
| Enlarged Cardiomediastinum | 0.1153 | 0.2525 | **0.4604** | −0.2297 | 0.0729 | 0.1785 | 0.0451 | 0.1007 |
| Cardiomegaly | 0.0101 | −0.0085 | 0.0085 | 0.0118 | 0.0098 | −0.0142 | −0.0135 | 0.0331 |
| Lung Opacity | 0.0265 | 0.0693 | **0.1272** | −0.0740 | 0.0166 | 0.0540 | 0.1064 | −0.0732 |
| Lung Lesion | 0.0216 | 0.0603 | 0.0302 | 0.0130 | −0.0051 | −0.0337 | −0.0238 | 0.0135 |
| Edema | 0.0189 | 0.0993 | 0.0092 | 0.0288 | 0.0149 | 0.0871 | 0.0554 | −0.0257 |
| Consolidation | 0.0118 | 0.0319 | 0.0441 | −0.0205 | −0.0092 | −0.0076 | −0.0054 | −0.0130 |
| Pneumonia | 0.0160 | −0.0305 | 0.0400 | −0.0080 | −0.0062 | −0.0129 | 0.0000 | −0.0124 |
| Atelectasis | 0.0034 | 0.0226 | 0.0171 | −0.0102 | 0.0038 | 0.0318 | 0.0170 | −0.0094 |
| Pneumothorax | 0.0253 | 0.0202 | 0.0529 | −0.0022 | −0.0102 | −0.0339 | −0.0204 | 0.0000 |
| Pleural Effusion | 0.0112 | 0.0086 | 0.0103 | 0.0122 | 0.0130 | 0.0401 | 0.0393 | −0.0134 |
| Pleural Other | −0.0019 | −0.0074 | −0.0044 | 0.0004 | 0.0127 | 0.0404 | 0.0250 | 0.0004 |
| Fracture | 0.0003 | 0.0000 | 0.0000 | 0.0005 | 0.0010 | 0.0000 | 0.0000 | 0.0019 |
| Support Devices | 0.0582 | 0.2354 | **0.4093** | −0.2929 | 0.0385 | 0.1802 | 0.2323 | −0.1553 |
| No Finding | 0.1693 | 0.0361 | **0.4329** | −0.0942 | 0.2800 | 0.3018 | 0.7067 | −0.1468 |
| **MACRO AVERAGE** | **0.0348** | **0.0564** | **0.1169** | **−0.0475** | **0.0309** | **0.0579** | **0.0831** | **−0.0214** |

> **Key Insight**: Using Qwen3.5 as the labeler (instead of CheXbert) generally yields higher Recall but slightly lower Specificity. The LLM-based labeler is more sensitive to positive findings mentioned in the generated reports.

## 🔧 Supported Models

| Model | `--model_name` | `--model_type` |
|-------|---------------|----------------|
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | `auto` |
| Qwen2.5-VL-72B | `Qwen/Qwen2.5-VL-72B-Instruct` | `auto` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `qwen3vl` |
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` | `auto` |

## 📝 Notes

- The CheXbert evaluation uses the **U-zeros** strategy: only explicit positive (1.0) is treated as positive; everything else (NaN, 0, -1/uncertain) is treated as negative.
- For the LLM-as-labeler, the same strategy is applied: only explicit positive assertions are labeled as 1.
- BERTScore uses `roberta-large` by default (downloaded from HuggingFace). You can specify a local model path with `--bertscore_model`.
- The toolkit automatically handles multi-GPU inference via `device_map="auto"`.

### ⚠️ RadGraph Compatibility

RadGraph depends on an older version of `transformers` (typically `<=4.12.x`). It may conflict with the newer `transformers` version required by Qwen models. **It is strongly recommended to create a separate conda/venv environment for RadGraph evaluation:**

```bash
# Create a separate environment for RadGraph
conda create -n radgraph_env python=3.8 -y
conda activate radgraph_env
pip install radgraph
# Run RadGraph evaluation in this environment
python evaluate_nlg.py \
    --llm_output ./results/qwen_output.json \
    --annotation_json ./data/annotation.json \
    --output_dir ./results/nlg_eval \
    --metrics radgraph
```

If you encounter errors like `ImportError` or version conflicts with `transformers`, this is the expected solution. Other metrics (BLEU, ROUGE, METEOR, BERTScore, RaTEScore) work fine with the latest `transformers`.

## 📄 License

This project uses CheXbert which is subject to its own license. See `CheXbert/` for details.

## 🙏 Acknowledgments

- [CheXbert](https://github.com/stanfordmlgroup/CheXbert) for clinical label extraction
- [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL) for vision-language models
- [RadGraph](https://github.com/jbdel/RadGraph) for radiology entity extraction
