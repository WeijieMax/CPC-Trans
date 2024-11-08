# CPC-Trans

Code for the MICCAI 2022 (early accepted) paper: "[Toward Clinically Assisted Colorectal Polyp Recognition via Structured Cross-modal Representation Consistency](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_14)"

---
<p align="center">
  <img align="middle" src="./assets/overview.png" style="width:80%" alt="The main figure"/>
</p>

## Dataset
We only provide the public part [[link]](https://drive.google.com/drive/folders/1e2t5HhQf08sTAE_CPRNVgpi6YUKgQSHn?usp=sharing) with its bbox we labeled [[link]](https://drive.google.com/file/d/1K06-VFm6b64Rhu-ehBtJ4OY6Yk7YZyIm/view?usp=sharing) of the CPC-paired Dataset while the another part is private. Thanks for understanding. And you should label the bounding box of the polyp and crop for preprocessing.

## Installation
```bash
pip install -r requirements.txt
```

## Download Pretrained Weights
We provide the pretrained weights of the model which were mentioned in the paper.
* Base Weights (vit_small_patch16_224_in21k) [link](https://drive.google.com/file/d/1I10_qXlUEtSjlvRWQjoBnKnVSLUYjF8C/view?usp=sharing)

For other model sizes, please refer to this table and download on the Internet or contact us to provide.
<p align="center">
  <img align="middle" src="./assets/variants.png" style="width:50%" alt="The main figure"/>
</p>

* Smaller Weights (vit_tiny_patch16_224_in21k)
* Larger Weights (vit_base_patch16_224_in21k)


## Run
we set the argument "fold" (default: 0) for k-fold cross validation, you can omit it if unnecessary.
```bash
python main.py --data_path /the/data/path/ --weights /the/pretrained/weights/path/
```


## Citation
If you use any part of this code and pretrained weights for your own purpose, please cite our paper.

```latex
@InProceedings{Ma-CPC-Trans,
  author={Ma, Weijie and Zhu, Ye and Zhang, Ruimao and Yang, Jie and Hu, Yiwen and Li, Zhen and Xiang, Li},
  title={Toward Clinically Assisted Colorectal Polyp Recognition via Structured Cross-Modal Representation Consistency},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
  year={2022},
  publisher={Springer Nature Switzerland},
  pages={141--150}
}
```

## Contact for Issues
- [Weijie Ma](https://WeijieMax.github.io/)
