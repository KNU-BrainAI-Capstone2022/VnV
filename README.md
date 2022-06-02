# View and Vision

## Setup Dataset
```bash
bash setup_dataset.sh [data_dir] [dataset : {voc2012,cityscapes}]
```
## Train model
```bash
python train.py -h
usage: train.py [-h] [--dataset DATASET] [--model MODEL] [-j NUM_WORKERS] [-b BATCH_SIZE] [--image-size IMAGE_SIZE] [--epochs EPOCHS] [--optim {sgd,adam}]
                [--lr LR] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--resume RESUME] [--test-only] [--test-model TEST_MODEL]

PyTorch Segmentation Training

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name
  --model MODEL         model name
  -j NUM_WORKERS, --num_workers NUM_WORKERS
                        number of data loading workers (default: 0)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        images per gpu
  --image-size IMAGE_SIZE
                        input image size
  --epochs EPOCHS       number of total epochs to run
  --optim {sgd,adam}    optimizer
  --lr LR               initial learning rate
  --momentum MOMENTUM   momentum
  --weight-decay WEIGHT_DECAY
                        weight_decay
  --resume RESUME       path of checkpoint
  --test-only           Only test the model
  --test-model TEST_MODEL
                        test model select

```
