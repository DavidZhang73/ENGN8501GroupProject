# Video Matting with Convolutional LSTM

Group Project Repo for ANU ENGN8501 21S2

## Description

We proposed a new DeepLab-based end-to-end architecture with ConvLSTM modules for video matting. 
The main challenge is alleviating the negative impact of removing pre-captured background from input. 
To achieve this goal, we add three ConvLSTM modules into the model to obtain temporal features to re-construct background from previous frames. 
We proved and showed how the ConvLSTM module help the inference.

## Related Repositories

- [BackgroundMattingV2 (BM2)](https://github.com/PeterL1n/BackgroundMattingV2)
- [ConvLSTM_pytorch (ConvLSTM)](https://github.com/ndrplz/ConvLSTM_pytorch)
- [RobustVideoMatting (RVM)](https://github.com/PeterL1n/RobustVideoMatting)

## Project Structure

```
├── dataset
│ ├── augmentation.py - Adapted from RVM by Jiahao Zhang
│ └── video_matte.py - Adapted from RVM by Jiahao Zhang
├── model
│ ├── convLSTM.py Adapted - from ConvLSTM by Hang Zhang
│ ├── decoderConvLSTM.py - Adapted from ConvLSTM by Hang Zhang
│ ├── decoder.py - Adapted from BM2 by Jiahao Zhang
│ ├── mobilenet.py - From BM2
│ ├── model.py - Adapted from BM2 by Jiahao Zhang
│ ├── refiner.py - From BM2
│ ├── resnet.py - Adapted from BM2 by Jiahao Zhang
│ └── utils.py from BM2
├── report - By Hang Zhang, Peng Zhang, Jiahao Zhang
├── scripts
│ ├── extract_frames.py - By Jiahao Zhang
│ └── server-setup.sh - By Jiahao Zhang
├── analysis.ipynb - By Peng Zhang
├── data_path.py - Adapted from BM2 by Jiahao Zhang
├── train.py - Adapted from BM2 by Jiahao Zhang
├── validate.py - By Peng Zhang
└── README.md - By Jiahao Zhang
```

## Usage

### Datasets

- [videomatte240k](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)
- [Background](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)

> modify data path in data_path.py

### Requirements

```shell
pip install -r requirements.txt
```

### Training

```shell
python train.py \
  --dataset-name videomatte8k \
  --model-backbone resnet50 \
  --model-name <MODEL_NAME> \
  --model-pretrain-initialization <MODEL_PRETRAIN_INITIALIZATION> \
  --model-last-checkpoint <MODEL_LAST_CHECKPOINT> \
  --batch-size 4 \
  --seq-length 8 \
  --num-workers 0 \
  --epoch-start 0 \
  --epoch-end 10 \
  --log-train-loss-interval 1 \
  --log-train-images-interval 20 \
  --log-valid-interval 1000 \
  --checkpoint-interval 1000
```

### Validation

```shell
python validate.py \
  --dataset-name videomatte8k \
  --model-backbone resnet50 \
  --model-checkpoint <MODEL_CHECKPOINT> \
  --output-path <OUTPUT_PATH> \
  --seq-length 1 \
  --num-workers 0
```

### Analysis

See code snippets in `analysis.ipynb`.

## Project members

- [Jiahao Zhang](https://github.com/DavidZhang73)
- [Hang Zhang](https://github.com/LeoZHANGboy)
- [Peng Zhang](https://github.com/Harley-ZP)

## License

This work is licensed under the MIT License.
