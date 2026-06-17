# RVQ-Grid

**Improving Audio Codec-based Speech Separation by Stacking Residual Vector Quantization Layers**

Accepted at **Interspeech 2026**.

- **Authors**: Nhu Minh Phuong Dinh, Roland Hartanto, Koichi Shinoda
- **Institution**: Institute of Science Tokyo, Japan
- **Link to the paper**: TBD

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

Code will be released soon.

## BibTeX

TBD.
