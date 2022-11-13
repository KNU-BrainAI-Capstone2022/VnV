import cv2
import numpy as np
import argparse
import os
import sys
from PIL import Image
from utils.Util import mask_colorize
import time

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

def preprocess(image):
    # Mean normalization
    # mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    # stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    # data = (np.asarray(image).astype('float16') / float(255.0) - mean) / stddev
    data=np.asarray(image).astype('float16')/255.0
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

def postprocess(input_file):
    num_classes = 19
    data = np.argmax(data,axis=0)
    print(data.shape)
    img = mask_colorize(data,getcmap()).astype(np.uint8)
    print(img.shape)

    return img

if __name__=='__main__':
    kargs = vars(get_args())
    TRT_LOGGER = trt.Logger()
    engine_file = kargs['engine']
    input_file = "video/test.jpg"
    output_file = 'test_out.jpg'

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
    out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,2*frame_height))
    print(f'{input_video} encoding ...')
    
    # print("Running TensorRT e for deeplabv3plut-ResNet50")
    start = time.time()
    total_frame =0
    with load_engine(engine_file) as engine:
        with engine.create_execution_context() as context:
            # Set input shape based on image dimensions for inference
            context.set_binding_shape(engine.get_binding_index("inputs"), (1, 3, frame_height, frame_width))
            while total_frame <= 30:
                ret, frame = cap.read()
                if not ret:
                    print('cap.read is failed')
                    engine.__del__()
                    break
                frame = cv2.resize(frame,(frame_width,frame_height))
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame = np.ascontiguousarray(frame)
                input_image = preprocess(frame)

                bindings = []
                for binding in engine:
                    binding_idx = engine.get_binding_index(binding)
                    size = trt.volume(context.get_binding_shape(binding_idx))
                    dtype = trt.nptype(engine.get_binding_dtype(binding))
                    if engine.binding_is_input(binding):
                        input_buffer = np.ascontiguousarray(input_image)
                        input_memory = cuda.mem_alloc(input_image.nbytes)
                        bindings.append(int(input_memory))
                    else:
                        output_buffer = cuda.pagelocked_empty(size, dtype)
                        output_memory = cuda.mem_alloc(output_buffer.nbytes)
                        bindings.append(int(output_memory))

                stream = cuda.Stream()
                # Transfer input data to the GPU.
                cuda.memcpy_htod_async(input_memory, input_buffer, stream)
                # Run inference
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                # Transfer prediction output from the GPU.
                cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
                # Synchronize the stream
                stream.synchronize()

                img = np.reshape(output_buffer,(19,frame_height,frame_width))
                img = np.argmax(img,axis=0)
                img = mask_colorize(img,getcmap()).astype(np.uint8)
                
                img = cv2.addWeighted(frame,0.3,img,0.7,0)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out_cap.write(img)
                total_frame +=1

    print(f'finish encoding - {out_name}')
    total_time = time.time()-start
    print(f'total frame = {total_frame} \ntotal time = {total_time:.2f}s')
    print(f'average time = {total_time/total_frame:.2f}')
    engine.__del__()
    