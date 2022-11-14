import cv2
import numpy as np
import argparse
import os
import sys
from utils.Util import mask_colorize
import time
import models
import torch
import torchvision.transforms.functional as F

try:
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
except ImportError:
    print("Failed to load tensorrt, pycuda")
    exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Video Encoding")
    # model option
    parser.add_argument("--engine", type=str, default="checkpoint/deeplabv3plus_resnet50_cityscapes/model_best_jetson_fp16.engine",help="model weights path")
    parser.add_argument("--dtype", type=str, choices=['fp32','fp16','int8'], default='fp16',help="weight dtype")
    # Dataset Options
    parser.add_argument("--video", type=str, help="input video name",required=True)
    parser.add_argument("--torch", action='store_true', help="Using torch deeplabv3+_mobilenet model for inference")
    return parser.parse_args()

def getcmap():
    # cityscapes cmap
    train_cmap = [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32)
    ]
    return train_cmap

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess(data):
    # Mean normalization
    # mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    # stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    # data = (np.asarray(data).astype('float16') / float(255.0) - mean) / stddev
    data=np.asarray(data).astype('float16')/255.0
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

if __name__=='__main__':
    kargs = vars(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kargs['torch']:
        model = models.model.__dict__["deeplabv3plus_mobilenet"](num_classes=19,output_stride=16,pretrained_backbone=False).to(device)
    else:
        TRT_LOGGER = trt.Logger()
        engine_file = kargs['engine']
    # cmap load
    cmap = getcmap()

    # --------------------------------------------
    # video info check
    # --------------------------------------------
    input_video = 'video/'+kargs['video']+'.mp4'
    if not os.path.exists('./video'):
        os.mkdir('./video')
    if not os.path.exists(input_video):
        print('input video is not exist\n')
        exit(1)
    cap = cv2.VideoCapture(input_video)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'video ({frame_width},{frame_height}), {fps} fps')

    # ----------------------------------------------
    # video write
    # ----------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_name = 'video/'+kargs['video']+'_'+kargs['dtype']+'_output.mp4'
    out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,frame_height))
    print(f'{input_video} encoding ...')
    
    # print("Running TensorRT e for deeplabv3plut-ResNet50")
    if kargs['torch']:
        model.eval()
        total_frame=0
        only_infer_time = 0
        with torch.no_grad():
            start = time.time()
            while total_frame <30:
                ret, frame = cap.read()
                if not ret:
                    print('cap.read is failed')
                    break
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                predict = F.to_tensor(frame).unsqueeze(0).to(device, dtype=torch.float32)
                only_run = time.time()
                predict = model(predict)
                only_infer_time += time.time()-only_run
                predict = predict.detach().argmax(dim=1).squeeze(0).cpu().numpy()
                predict = mask_colorize(predict,cmap).astype(np.uint8)
                
                result = cv2.addWeighted(frame,0.3,predict,0.7,0)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                
                out_cap.write(result)
                total_frame +=1

    else:
        # load engine
        with load_engine(engine_file) as engine:
            # engine create
            with engine.create_execution_context() as context:
                # Set input shape based on image dimensions for inference
                context.set_binding_shape(
                    engine.get_binding_index("inputs"),
                    (1, 3, frame_height, frame_width)
                    )
                # input_image= np.zeros((540,960,3)).astype('float32')
                # time setting
                start = time.time()
                total_frame =0
                only_infer_time = 0
                # read video
                while total_frame < 30:
                    ret, frame = cap.read()
                    if not ret:
                        print('cap.read is failed')
                        engine.__del__()
                        break
                    frame = cv2.resize(frame,(frame_width,frame_height))
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    # Using numpy 
                    # input_image = np.ascontiguousarray(frame)
                    input_image = preprocess(frame)

                    # Using tensor
                    # input_image = F.to_tensor(frame)
                    bindings = []
                    for binding in engine:
                        # find binding index
                        binding_idx = engine.get_binding_index(binding)
                        # memory volum set 
                        size = trt.volume(context.get_binding_shape(binding_idx))
                        # memory type set
                        dtype = trt.nptype(engine.get_binding_dtype(binding))
                        if engine.binding_is_input(binding):
                            input_buffer = np.ascontiguousarray(input_image)
                            # memory alloc 
                            input_memory = cuda.mem_alloc(input_image.nbytes)
                            bindings.append(int(input_memory))
                        else:
                            output_buffer = cuda.pagelocked_empty(size, dtype)
                            output_memory = cuda.mem_alloc(output_buffer.nbytes)
                            bindings.append(int(output_memory))
                    
                    # generate cuda class
                    stream = cuda.Stream()
                    # Transfer input data to the GPU.
                    cuda.memcpy_htod_async(input_memory, input_buffer, stream)
                    # Run inference
                    only_run = time.time()
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    only_infer_time += time.time()-only_run

                    # Transfer prediction output from the GPU.
                    cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
                    # Synchronize the stream
                    stream.synchronize()

                    # Using tensor
                    # img = torch.from_numpy(output_buffer)
                    # img = torch.reshape(img,(19,frame_height,frame_width))
                    # predict = predict.detach().argmax(dim=0).cpu().numpy()
                    # img = mask_colorize(predict,cmap).astype(np.uint8)
                    
                    # Using numpy 
                    # ------------------------------ 
                    img = np.argmax(np.reshape(output_buffer,(19,frame_height,frame_width)),axis=0)
                    img = mask_colorize(img,cmap).astype(np.uint8)
                    
                    img = cv2.addWeighted(frame,0.3,img,0.7,0)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out_cap.write(img)
                    total_frame +=1
        engine.__del__()

    print(f'finish encoding - {out_name}')
    total_time = time.time()-start
    print(f'total frame = {total_frame} \ntotal time = {total_time:.2f}s')
    print(f'average time = {total_time/total_frame:.2f}s')
    print(f'Only inference time : {only_infer_time:.2f}s')
    