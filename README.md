# View and Vision

## Setup Dataset
```bash
bash setup_dataset.sh [data_dir] [dataset : {voc2012,cityscapes}]
```
## Train model
```
python train.py -h
usage: train.py [-h] --data_root DATA_ROOT --dataset {voc2012,cityscapes}
                [--num_classes NUM_CLASSES] --model
                {deeplabv3_resnet101,deeplabv3_resnet50,deeplabv3plus_resnet101,deeplabv3plus_resnet50}
                [--output_stride {8,16}] [--test_only] [--save_results]
                [--total_iters TOTAL_ITERS] [-j NUM_WORKERS] [--batch_size BATCH_SIZE]
                [--val_batch_size VAL_BATCH_SIZE] [--lr LR] [--lr_scheduler {exp,step}]
                [--step_size STEP_SIZE] [--weight_decay WEIGHT_DECAY]
                [--crop_size CROP_SIZE] [--resume] [--print_interval PRINT_INTERVAL]
                [--val_interval VAL_INTERVAL]

PyTorch Segmentation Training

optional arguments:
  -h, --help            show this help message and exit
  --data_root DATA_ROOT
                        path to Dataset
  --dataset {voc2012,cityscapes}
                        dataset name
  --num_classes NUM_CLASSES
                        num classes (default: None)
  --model {deeplabv3_resnet101,deeplabv3_resnet50,deeplabv3plus_resnet101,deeplabv3plus_resnet50}
                        model name
  --output_stride {8,16}
  --test_only           Only test the model
  --save_results        save segmentation results
  --total_iters TOTAL_ITERS
                        number of total iterations to run (default: 30k)
  -j NUM_WORKERS, --num_workers NUM_WORKERS
                        number of data loading workers (default: 0)
  --batch_size BATCH_SIZE
                        images per gpu
  --val_batch_size VAL_BATCH_SIZE
                        images per gpu
  --lr LR               initial learning rate
  --lr_scheduler {exp,step}
                        learning rate scheduler policy
  --step_size STEP_SIZE
                        (default: 10k)
  --weight_decay WEIGHT_DECAY
                        weight_decay
  --crop_size CROP_SIZE
                        input image crop size
  --resume
  --print_interval PRINT_INTERVAL
                        print interval of loss (default: 10)
  --val_interval VAL_INTERVAL
                        iteration interval for eval (default: 100)

```
