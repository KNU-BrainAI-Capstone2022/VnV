# View and Vision
Goal : Real-Time Semantic Segmentation on Edge Devices or Mobile Devices


## Setup Dataset
```bash
bash setup_dataset.sh [data_dir] [dataset : {voc2012,cityscapes}]
```
The selected dataset is installed under data_dir. When you install cityscapes, you must have an account at https://www.cityscapes-dataset.com/.  
Required Space  
Pascal VOC 2012 : 2GB  
Cityscapes : 11.6 GB  
## Train model
```bash
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

Train Example
```bash
python train.py --dataset cityscapes --model deeplabv3_resnet50
```
Continue Train Example
```bash
python train.py --dataset cityscapes --model deeplabv3_resnet50 --resume
```
Evaluate Example
```bash
# Given the --save_results option, the Segmentation results images are stored in the ./results folder.
python train.py --dataset cityscapes --model deeplabv3_resnet50 --test_only --save_results
```
### Encoding MP4
When the train is completed above, insert the mp4 file to be segmented in the ./video folder and execute the code. or you can use a pretrained model.
available pretrained model : https://drive.google.com/drive/folders/1xG5QCrPuSSFUFPGfVhr7qE5OkVlJii3u?usp=sharing
```bash
python encoding.py -h
usage: encoding.py [-h] --model MODEL --input INPUT [--pair] [--test]

PyTorch Segmentation Video Encoding

optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  model name
  --input INPUT  input video name
  --pair         Generate pair frame
  --test         Generate thunbnail
```
