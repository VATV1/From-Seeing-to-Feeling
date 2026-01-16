# From Seeing to Feeling: Visual Affordance Guided Softness Verification via Active Touching for Fabric-like Deformable Object Manipulation

## TODO

- [✔] Release the code of OS-VAL.
- [✔] Release the code of AVTNet.
- [✔] Release the DOVT dataset.
- [x] Paper (Under Review).

## Usage
### 1.Requirements
  Code is tested under Pytorch 2.0.0, python 3.10, and CUDA 12.1 
### 2.Dataset
  Download our proposed DOVT dataset from [Google Drive](https://drive.google.com/drive/folders/1gN2t7nQUk-_fB5gUaljWAj0KF0KWqmji).
  Download the AGDDO15 dataset from [OS-AGDO](https://github.com/Dikay1/OS-AGDO).
  Download the TAG dataset from [Google Drive](https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B).
#### OS-VAL
  Put the AGDDO15 dataset in the `OS-VAL/data` folder with the following structure:  
```
AGDDO15 
├── one-shot-seen
└── Seen
```
#### AVTNet
  Put the DOVT and TAG datasets in the `AVTNet/data` folder with the following structure:  
```
DOVT 
├── train
└── test
TAG 
├── train
└── test
```
### 3.Train and Test
  Run following commands to start training or testing:

```
python train.py
python test.py --model_file <PATH_TO_MODEL>
```
 
