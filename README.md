# Multimodal-Emotion-Recognition-using-AVTCA

This repository implements a multimodal network for emotion recognition using the Audio-Video Transformer Fusion with Cross Attention (AVT-CA) model, as given in the paper [Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention](https://arxiv.org/pdf/2407.18552). The implementation supports the RAVDESS dataset, which includes speech and frontal face view data across 8 distinct emotions: 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, and 08 = surprised.

<p align="center">
<img src="https://github.com/shravan-18/AVTCA/blob/main/img/AVTCA.png" alt="drawing" height="70%"/>
</p>
<p align = "center">
AVT-CA Model Diagram
</p>

## Citation

If you use our work, please cite as:
```bibtex
@article{chumachenko2022self,
  title={Self-attention fusion for audiovisual emotion recognition with incomplete data},
  author={Chumachenko, Kateryna and Iosifidis, Alexandros and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:2201.11095},
  year={2022}
}
```

If you are referencing our work, please also cite the following related paper:

**Chumachenko, K., Iosifidis, A., & Gabbouj, M. (2022).** *Self-attention fusion for audiovisual emotion recognition with incomplete data*. arXiv. https://arxiv.org/abs/2201.11095

## References

This work incorporates EfficientFace, available at [EfficientFace GitHub repository](https://github.com/zengqunzhao/EfficientFace). Please cite the paper titled "Robust Lightweight Facial Expression Recognition Network with Label Distribution Training" if you use EfficientFace. We appreciate @zengqunzhao for providing both the implementation and the pretrained model for EfficientFace!

The training pipeline code has been adapted from [Efficient-3DCNNs GitHub repository](https://github.com/okankop/Efficient-3DCNNs), which is licensed under the MIT license. Additionally, parts of the fusion implementation are based on the [timm library](https://github.com/rwightman/pytorch-image-models), available under the Apache 2.0 license. For data preprocessing, we utilized [facenet-pytorch](https://github.com/timesler/facenet-pytorch).
