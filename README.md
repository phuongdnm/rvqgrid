# RVQ-Grid

**Improving Audio Codec-based Speech Separation By Stacking Residual Vector Quantization Layers**

Published in **Interspeech 2026**.

- **Authors**: Nhu Minh Phuong Dinh, Roland Hartanto, Koichi Shinoda
- **Institution**: Institute of Science Tokyo, Japan
- **Paper**: [ISCA Archive](https://www.isca-archive.org/interspeech_2026/dinh26_interspeech.html)
- **DOI**: [10.21437/Interspeech.2026-2296](https://doi.org/10.21437/Interspeech.2026-2296)

## Abstract

Speech separation models trained on raw waveforms achieve high performance but come with substantial
computational costs. Neural Audio Codecs (NACs) offer an efficient alternative by operating in compressed latent spaces. However, the existing codec-based method aggregates per-layer Residual Vector Quantization
(RVQ) vectors into a single vector, collapsing the coarse-to-fine acoustic hierarchy. We propose
**RVQ-Grid** to explicitly preserve this hierarchy by stacking per-layer RVQ vectors into a 3D grid and
processing it using dual-axis recurrent blocks. On WSJ0-2Mix, RVQ-Grid achieves a +3.6 dB SI-SDRi
improvement over the prior codec-based method. Our evaluation on the Automatic Speech Recognition task
shows that RVQ-Grid achieves an 8.6% WER, demonstrating comparable performance to the SepFormer baseline
under the same codec condition. During inference, RVQ-Grid achieves a 6x reduction in MACs compared to
SepFormer.

## Code

### Environment

- Python 3.11
- PyTorch 2.7.1
- CUDA 11.8

```bash
conda create -n rvqgrid python=3.11
conda activate rvqgrid
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

### Installation

```bash
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
git checkout v1.0.0

pip install -r requirements.txt
pip install -e .
pip install descript-audio-codec encodec
pip install pandas transformers
```

### Copy the RVQ-Grid files into SpeechBrain

```bash
cp src/models/*.py speechbrain/speechbrain/lobes/models/
cp src/train_rvq_gridnet*.py speechbrain/recipes/WSJ0Mix/separation/
cp src/hparams/*.yaml speechbrain/recipes/WSJ0Mix/separation/hparams/
```

### Training

```bash
# In speechbrain/recipes/WSJ0Mix/separation/
# 1 GPU
python train_rvq_gridnet.py hparams/RVQ_GridNet_DAC.yaml

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_rvq_gridnet_encodec.py hparams/RVQ_GridNet_EnCodec.yaml
```

## BibTeX

```bibtex
@inproceedings{dinh26_interspeech,
  title     = {{Improving Audio Codec-based Speech Separation By Stacking Residual Vector Quantization Layers}},
  author    = {Nhu Minh Phuong Dinh and Roland Hartanto and Koichi Shinoda},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {7153--7157},
  doi       = {10.21437/Interspeech.2026-2296},
  issn      = {2958-1796},
}
```
