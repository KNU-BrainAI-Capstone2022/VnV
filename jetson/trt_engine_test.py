import sys
sys.path.append('../')
import cv2
import numpy as np
import argparse
import os
from utils.Util import mask_colorize
import time
import torch
import torchvision.transforms.functional as F
from utils.Dataset import CustomCityscapesSegmentation
from models.model import deeplabv3plus_mobilenet
from torch2trt import TRTModule
import onnx
import onnxruntime

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
    parser.add_argument("--engine", type=str, default="../checkpoint/deeplabv3plus_mobilenet_cityscapes/model_best_jetson_fp16.engine",help="model engine path")
    parser.add_argument("--onnx", type=str, default="../checkpoint/deeplabv3plus_mobilenet_cityscapes/model_best_jetson.onnx", help="model onnx path")
    parser.add_argument("--base", type=str, default="../checkpoint/deeplabv3plus_mobilenet_cityscapes/model_best.pth", help="Base model torch (.pth)")
    parser.add_argument("--dtype", type=str, choices=['fp32','fp16','int8'], default='fp16',help="weight dtype")
    # Dataset Options
    parser.add_argument("--video", type=str, default="../video/220619_2.mp4",help="input video name")
    parser.add_argument("--torch", action='store_true', help="Using torch deeplabv3+_mobilenet model for inference")
    parser.add_argument("--test", action="store_true", help="testing create builder")
    parser.add_argument("--torch2trt", action="store_true", help="Using torch2trt module")
    return parser.parse_args()

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess(data):
    data=np.asarray(data).astype('float32')/255.0
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    def __init__(self,engine_path,dtype=np.float32):
        print(f"\nTRT Engine init...")
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # builder
        # self.builder = trt.Builder(self.logger)
        # self.network = builder.cerate_network()
        # self.config = builder.create_builder_config()
        self.runtime = trt.Runtime(self.logger)
    
        # Load engine
        self.engine = self.load_engine(self.runtime, self.engine_path)

        # memory 할당
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
        
        # print(self.context.get_binding_shape(0))
        # print(self.context.get_binding_shape(1))
        print(f"engine.get_location -> {self.engine.get_location(0)}")
        print(f"engine.get_binding_dtype -> {self.engine.get_binding_dtype}\n")

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):

        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            print(f"  {binding}")
            size = tuple(self.engine.get_binding_shape(binding))
            print(f"  binding size : {size}")
            dtype = self.engine.get_binding_dtype(binding)
            print(f"  binding dtype : {dtype}")
            location = self.engine.get_location(binding)
            print(f"  binding location : {location}\n")

            # # np.ndarray의 pagelocked를 할당
            # host_mem = cuda.pagelocked_empty(size, self.dtype)
            host_mem = torch.empty(size=size, dtype=torch.float32, device=torch.device("cuda"))
            # device memory 할당
            device_mem = host_mem.data_ptr()
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs = host_mem
            else:
                outputs = host_mem
        
        return inputs, outputs, bindings, stream

    def __call__(self,x,batch_size=1):
        
        # -------------------------- torch tensor
        # x = x.contiguous()
        # self.inputs[0].host = x.ravel()

        # --------------------------- numpy
        # x = x.astype(self.dtype).ravel()
        # x = np.ascontiguousarray(x)
        
        # # x.ravel is Returns to a continuous 1-dimensional plane
        # np.copyto(self.inputs[0].host,x.ravel())
        # ----------------------------------------------
        self.bindings[0] = x.contiguous().data_ptr()
        # [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # infer time check
        only_run = time.time()
        check = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        i_time = time.time()-only_run

        # [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()

        return self.outputs , i_time

    def __del__(self):
        if self.engine is not None:
            del self.engine

def lib_version():
    # os.environ['CUDA_LAUNCH_BLOCKING']='1'
    print(f"\ntrt version : {trt.__version__}")
    print(f"torch version : {torch.__version__}")
    print(f"onnx verison : {onnx.__version__}")
    print(f"onnxruntime version : {onnxruntime.__version__}")


if __name__=='__main__':
    kargs = vars(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # lib_version()
    if kargs['torch']:
        model = deeplabv3plus_mobilenet(num_classes=19,output_stride=16,pretrained_backbone=False).to(device)
        model_path = kargs['base']
        print(f"{model_path} model loading ...")
        model.load_state_dict(torch.load(model_path)['model_state'])
        model.eval()
    elif kargs['torch2trt']:
        model = TRTModule()
        model_path = kargs['base'].replace(".pth","_jetson_trt_fp16.pth")
        print(f"{model_path} model loading ....")
        model.load_state_dict(torch.load(model_path))
    else:
        TRT_LOGGER = trt.Logger()
        engine_path = kargs['engine']
        print(f"{engine_path} engine loading ...")
    # cmap load
    cmap = CustomCityscapesSegmentation.cmap
    print("Model Loading Done.")
    # version print()
    
    # version print()
    # --------------------------------------------
    # video info check
    # --------------------------------------------
    if not os.path.exists(kargs['video']):
        print('input video is not exist\n')
        exit(1)
    cap = cv2.VideoCapture(kargs['video'])
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'video ({frame_width},{frame_height}), {fps} fps')

    # ----------------------------------------------
    # video write
    # ----------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_name = kargs['video'][:-4]+'_output.mp4'
    out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,frame_height))
    print(f"{kargs['video']} encoding ...")

    # if kargs['test']:
    #     create_builder = Load_engine(onnx_path=kargs['onnx'])
    #     e, l = create_builder.parse_or_load()
    #     exit(1)
    
    
    if kargs['torch'] or kargs['torch2trt']:
        print("Running Pytorch\n")
        total_frame=0
        only_infer_time = 0
        with torch.no_grad():
            start = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('cap.read is failed')
                    break
                
                frame = cv2.resize(frame, (frame_width,frame_height))
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                input_image = F.to_tensor(frame).to(device).unsqueeze(0)
                # input_image = F.to_tensor(frame).unsqueeze(0)
                # print(f"total frame : {total_frame}")
                only_run = time.time()
                predict = model(input_image)
                only_infer_time += time.time()-only_run
                
                #print(predict.shape)
                predict = predict.detach().squeeze(0).argmax(dim=0).cpu().numpy()
                predict = mask_colorize(predict,cmap).astype(np.uint8)
                
                result = cv2.addWeighted(frame,0.3,predict,0.7,0)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                
                out_cap.write(result)
                total_frame +=1
    else:
        model = TrtModel(engine_path)
        print("TRT Engine running...\n")
        start = time.time()
        total_frame =0
        only_infer_time = 0
        # read video
        while total_frame < 30:
            ret, frame = cap.read()
            if not ret:
                print('cap.read is failed')
                break
            total_frame +=1
            frame = cv2.resize(frame,(frame_width,frame_height))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # input_image = preprocess(frame)
            input_image = F.to_tensor(frame).cuda()

            outputs,t = model(input_image)

            only_infer_time +=t

            img = outputs.detach().squeeze(0).argmax(dim=0).cpu().numpy()
            img = mask_colorize(img,cmap).astype(np.uint8)

            # img = np.argmax(np.reshape(outputs[0],(19,frame_height,frame_width)),axis=0)
            # img = mask_colorize(img,cmap).astype(np.uint8)
            img = cv2.addWeighted(frame,0.3,img,0.7,0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            out_cap.write(img)
            # cv2.imwrite('../video/test.jpg',img)


        del(model)
    print(f'finish encoding - {out_name}')
    total_time = time.time()-start
    print(f'total frame = {total_frame} \ntotal time = {total_time:.2f}s')
    print(f'average time = {total_time/total_frame:.2f}s')
    print(f'Only inference time : {only_infer_time:.2f}s')
    
