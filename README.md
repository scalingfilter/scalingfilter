# ScalingFilter

[![arXiv](https://img.shields.io/badge/arXiv-2408.08310-b31b1b.svg)](https://arxiv.org/abs/2408.08310)

This is an official implementation for "ScalingFilter: Assessing Data Quality through Inverse Utilization of Scaling Laws", which is accepted by EMNLP 2024 Main.




## Getting Started

### Installation

To run the code, a conda environment with `PyTorch` installed is required:

```bash
conda create -n scalingfilter python=3.10
conda activate scalingfilter
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then, clone this repo and install dependencies:

```bash
git clone https://github.com/scalingfilter/scalingfilter.git
cd scalingfilter
pip install -e .
```



### ScalingFilter

To calculate *Quality Factors* on a single node:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=<num_gpus> \
  scalingfilter/quality_factor.py \
  --input_dir <input_dir> \
  --output_dir <output_dir>
```

We assume that the original data is stored in `input_dir` in JSONL format, and the `text` field of each sample contains the text data to be filtered. The JSONL files in `output_dir` will contain the original text data along with the perplexity scores and *quality factors* calculated by *meta-models*.

To calculate *Quality Factors* with multiple nodes:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=<num_gpus> \
  scalingfilter/quality_factor.py \
  --input_dir <input_dir> \
  --output_dir <output_dir> \
  --nparts <num_parts> \
  --rank_part_start <rank_part_start> \
  --rank_nparts <rank_nparts>
```

The `nparts` is the total number of parts, `rank_part_start` is the starting index of the parts for the current node, and `rank_nparts` is the number of parts for the current node to process. This design is to support distributed inference with multiple nodes equipped with different GPUs: nodes with better GPUs can be set to process more `rank_nparts`, and vice versa.


### Semantic Diversity

To calculate *Semantic Diversity* for a certain dataset:

```bash
python -m scalingfilter.semantic_diversity \
  --input_dir <input_dir> \
  --output_dir <output_dir>
```

The calculated *Semantic Diversity* will be displayed in the terminal and saved to `results.txt` within the `output_dir`.


## Citation

```
@inproceedings{li-etal-2024-scalingfilter,
    title = "{S}caling{F}ilter: Assessing Data Quality through Inverse Utilization of Scaling Laws",
    author = "Li, Ruihang  and
      Wei, Yixuan  and
      Zhang, Miaosen  and
      Yu, Nenghai  and
      Hu, Han  and
      Peng, Houwen",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.187",
    pages = "3209--3222",
    abstract = "High-quality data is crucial for the pre-training performance of large language models. Unfortunately, existing quality filtering methods rely on a known high-quality dataset as reference, which can introduce potential bias and compromise diversity. In this paper, we propose ScalingFilter, a novel approach that evaluates text quality based on the perplexity difference between two language models trained on the same data, thereby eliminating the influence of the reference dataset in the filtering process. An theoretical analysis shows that ScalingFilter is equivalent to an inverse utilization of scaling laws. Through training models with 1.3B parameters on the same data source processed by various quality filters, we find ScalingFilter can improve zero-shot performance of pre-trained models in downstream tasks. To assess the bias introduced by quality filtering, we introduce semantic diversity, a metric of utilizing text embedding models for semantic representations. Extensive experiments reveal that semantic diversity is a reliable indicator of dataset diversity, and ScalingFilter achieves an optimal balance between downstream performance and semantic diversity.",
}
```
