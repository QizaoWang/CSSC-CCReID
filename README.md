# Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification
> Official PyTorch implementation of ["Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification"](https://arxiv.org/abs/2405.16597).
>
> Accepted by ICASSP 2025
>
> Qizao Wang, Xuelin Qian, Bin Li, Lifeng Chen, Yanwei Fu, Xiangyang Xue
>
> Fudan University, Northwestern Polytechnical University


## Getting Started

### Environment

- Python == 3.8
- PyTorch == 1.12.1

### Prepare Data

Please download cloth-changing person re-identification datasets and place them in any path `DATASET_ROOT`:

    DATASET_ROOT
    	└─ LTCC-reID or PRCC or Celeb-reID
    		├── train
    		├── query
    		└── gallery


### Training

```sh
# LTCC
python main.py --gpu_devices 0 --dataset ltcc --dataset_root DATASET_ROOT --dataset_filename LTCC-reID --save_dir SAVE_DIR --save_checkpoint

# PRCC
python main.py --gpu_devices 0 --dataset prcc --dataset_root DATASET_ROOT --dataset_filename PRCC --save_dir SAVE_DIR --save_checkpoint

# Celeb-reID
python main.py --gpu_devices 0 --dataset celeb --dataset_root DATASET_ROOT --dataset_filename Celeb-reID --num_instances 4 --save_dir SAVE_DIR --save_checkpoint
```

`--dataset_root` : replace `DATASET_ROOT` with your dataset root path

`--save_dir`: replace `SAVE_DIR` with the path to save log file and checkpoints


### Evaluation

```sh
python main.py --gpu_devices 0 --dataset DATASET --dataset_root DATASET_ROOT --dataset_filename DATASET_FILENAME --resume RESUME_PATH --save_dir SAVE_DIR --evaluate
```

`--dataset`: replace `DATASET` with the dataset name

`--dataset_filename`: replace `DATASET_FILENAME` with the folder name of the dataset

`--resume`: replace `RESUME_PATH` with the path of the saved checkpoint

The above three arguments are set corresponding to Training.


### Results

- **Celeb-reID**

| Backbone  | Rank-1 | Rank-5 | mAP  |
| :-------: |:------:|:------:|:----:|
| ResNet-50 |  64.5  |  78.1  | 17.3 |

- **LTCC**

| Backbone  |    Setting     | Rank-1 | mAP  |
| :-------: | :------------: |:------:|:----:|
| ResNet-50 | Cloth-Changing |  43.6  | 18.6 |
| ResNet-50 |    Standard    |  78.1  | 40.2 |

- **PRCC**

| Backbone  |    Setting     | Rank-1 | mAP  |
| :-------: | :------------: |:------:|:----:|
| ResNet-50 | Cloth-Changing |  65.5  | 63.0 |
| ResNet-50 |    Standard    |  100   | 99.1 |

You can achieve similar results with the released code.

## Citation

Please cite the following paper in your publications if it helps your research:

```
@inproceedings{wang2025content,
  title={Content and salient semantics collaboration for cloth-changing person re-identification},
  author={Wang, Qizao and Qian, Xuelin and Li, Bin and Chen, Lifeng and Fu, Yanwei and Xue, Xiangyang},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1-5},
  year={2025},
  organization={IEEE}
}
```

The code is based on [FIRe-CCReID](https://github.com/QizaoWang/FIRe-CCReID):
```
@article{wang2024exploring,
  title={Exploring fine-grained representation and recomposition for cloth-changing person re-identification},
  author={Wang, Qizao and Qian, Xuelin and Li, Bin and Xue, Xiangyang and Fu, Yanwei},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={19},
  pages={6280-6292},
  year={2024},
  publisher={IEEE}
}
```


## Contact

Any questions or discussions are welcome!

Qizao Wang (<qzwang22@m.fudan.edu.cn>)