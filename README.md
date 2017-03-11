# Semantic Autoencoder for Zero-shot Learning, CVPR 2017.

## Abstract
Existing zero-shot learning (ZSL) models typically learn a projection function from a visual feature space to a semantic embedding space (e.g.~attribute space). However, such a projection function is only concerned with predicting the training seen class semantic representation (e.g.~attribute prediction) or classification. When applied to test data, which in the context of ZSL contains different (unseen) classes without training data, a ZSL model typically suffers from the project domain shift problem. In this work, we present a novel solution to ZSL based on learning a Semantic AutoEncoder (SAE). Taking the encoder-decoder paradigm, an encoder aims to project a visual feature vector into the semantic space as in the existing ZSL models. However, the decoder exerts an additional constraint, that is, the projection/code must be able to reconstruct the original visual feature. We show that with this additional reconstruction constraint, the learned projection function from the seen classes is able to generalise better to the new unseen classes. Importantly, the encoder and decoder are linear and symmetric which enable us to develop an extremely efficient learning algorithm. Extensive experiments on six benchmark datasets demonstrate that the proposed SAE outperforms significantly the existing ZSL models with the additional benefit of lower computational cost. Furthermore, when the SAE is applied to supervised clustering problem, it also beats the state-of-the-art.

## An implementation of SAE in MATLAB

## Download

Paper

## Cite

```
@ARTICLE{ekodirov_cvpr2017,
   author = {Elyor Kodirov, Tao Xiang, and Shagong Gong},
    title = "{Semantic Autoencoder for Zero-shot Learning}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1612.04642},
 primaryClass = "cs.CV",
 keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Learning, Statistics - Machine Learning},
     year = 2016,
    month = dec,
   adsurl = {http://adsabs.harvard.edu/abs/2016arXiv161204642W},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

