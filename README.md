# Official Implementation of AAAI'24 paper "Dirichlet-Based Prediction Calibration for Learning with Noisy Labels"

by **Chen-Chen Zong, Ye-Wen Wang, Ming-Kun Xie, Sheng-Jun Huang**

[[Main paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29672) [[Appendix]](https://github.com/chenchenzong/DPC/blob/main/AAAI2024_DPC_appendix.pdf) [[Code]](https://github.com/chenchenzong/DPC/blob/main/AAAI2024_DPC_code/README.md)

## Abstract

Learning with noisy labels can significantly hinder the generalization performance of deep neural networks (DNNs). Existing approaches address this issue through loss correction or example selection methods. However, these methods often rely on the model's predictions obtained from the softmax function, which can be over-confident and unreliable. In this study, we identify the translation invariance of the softmax function as the underlying cause of this problem and propose the *Dirichlet-based Prediction Calibration* (DPC) method as a solution. Our method introduces a calibrated softmax function that breaks the translation invariance by incorporating a suitable constant in the exponent term, enabling more reliable model predictions. To ensure stable model training, we leverage a Dirichlet distribution to assign probabilities to predicted labels and introduce a novel evidence deep learning (EDL) loss. The proposed loss function encourages positive and sufficiently large logits for the given label, while penalizing negative and small logits for other labels, leading to more distinct logits and facilitating better example selection based on a large-margin criterion. Through extensive experiments on diverse benchmark datasets, we demonstrate that DPC achieves state-of-the-art performance.


## Citation

If you find this repo useful for your research, please consider citing the paper.

```bibtex
@inproceedings{zong2024dirichlet,
  title={Dirichlet-Based Prediction Calibration for Learning with Noisy Labels},
  author={Zong, Chen-Chen and Wang, Ye-Wen and Xie, Ming-Kun and Huang, Sheng-Jun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={15},
  pages={17254--17262},
  year={2024}
}
```
## Acknowledgement

Thanks to Li et al. for publishing their code for [DivideMix](https://github.com/LiJunnan1992/DivideMix). Our implementation is heavily based on their work.
