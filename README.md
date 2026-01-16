# From Seeing to Feeling: Visual Affordance Guided Softness Verification via Active Touching for Fabric-like Deformable Object Manipulation

## TODO

- [✔] Release the code of OS-VAL.
- [✔] Release the code of AVTNet.
- - [✔] Release the code of AVTNet.

- [x] Release the code.
- [x] Release the [arxiv preprint](https://arxiv.org/pdf/2503.01092).

## Citation
If our work is helpful to you, please consider citing us by using the following BibTeX entry:

```
@article{jia2025one,
  title={One-Shot Affordance Grounding of Deformable Objects in Egocentric Organizing Scenes},
  author={Jia, Wanjun and Yang, Fan and Duan, Mengfei and Chen, Xianchi and Wang, Yinxi and Jiang, Yiming and Chen, Wenrui and Yang, Kailun and Li, Zhiyong},
  journal={arXiv preprint arXiv:2503.01092},
  year={2025}
}
```

## Usage
### 1.Requirements
  Code is tested under Pytorch 1.12.1, python 3.7, and CUDA 11.3 
  
```
pip install -r requirements.txt
```
### 2.Dataset
  Download the AGDDO15 dataset from [Baidu Pan]( https://pan.baidu.com/s/1KV4PrwBExB8A5MDq9ZxDgw?pwd=S7U2)[S7U2].(you can annotate your own one-shot data in the same format).
  
  Put the data in the `dataset` folder with the following structure:  
```
dataset 
├── one-shot-seen
└── Seen
```
### 3.Train and Test
  Run following commands to start training or testing:

```
python train.py
python test.py --model_file <PATH_TO_MODEL>
```
 
