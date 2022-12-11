import cv2
import numpy as np
import argparse
import os
import time
import torch
import torchvision.transforms.functional as F
from trt_model import WrappedModel

from utils.colormap import mask_colorize,cmap_cityscapes,cmap_voc

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
    parser = argparse.ArgumentParser(description="Video Segmentation Encoding Using Several Methods")
    # Model checkpoint option
    parser.add_argument("checkpoint", type=str, help="model checkpoint path")
    parser.add_argument('-n',"--num_classes", type=int, help="number of classes of model checkpoint")

    # Video Options
    parser.add_argument("--video", type=str, default='', help="input video name")
    parser.add_argument("--cam", action='store_true', help="input video name")
    
    # onnxrumtime option
    parser.add_argument("--ort", action='store_true', help="Using onnx model for inference")

    # torch option
    parser.add_argument("--torch", action='store_true', help="Using torch deeplabv3+_mobilenet model for inference")
    
    # tensorrt option
    parser.add_argument("--trt", action="store_true", help="Using tensorrt engine")
    
    # wrapped model
    parser.add_argument("--wrapped", action="store_true", help="wrapped model")

    return parser.parse_args()
    
def preprocess(data):
    data = np.asarray(data).astype('float32')/255.0
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

def func_torch_plain(model,frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = F.to_tensor(frame).unsqueeze(0).cuda()
        
    only_run = time.time()
    predict = model(frame)[0]
    only_infer_time = time.time()-only_run
    
    predict = predict.detach().argmax(dim=0).cpu().numpy()
    
    return predict, only_infer_time

def func_torch_wrapped(model,frame):
    frame = frame.unsqueeze(0).cuda()
        
    only_run = time.time()
    predict = model(frame)[0][0]
    only_infer_time = time.time()-only_run
    
    predict = predict.detach().cpu().numpy()

    return predict, only_infer_time

def func_trt_plain(model,frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = preprocess(frame)
    frame = np.expand_dims(frame,axis=0)
    
    only_run = time.time()
    predict = model(frame)[0]
    only_infer_time = time.time()-only_run
    
    predict = predict.argmax(axis=0)

    return predict, only_infer_time

def func_trt_wrapped(model,frame):
    frame = np.expand_dims(frame,axis=0)
    
    only_run = time.time()
    predict = model(frame)[0][0]
    only_infer_time = time.time()-only_run

    return predict, only_infer_time

def func_ort_plain(model,frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = preprocess(frame)
    frame = np.expand_dims(frame,axis=0)
    
    only_run = time.time()
    predict = model(frame)[0]
    only_infer_time = time.time()-only_run
    
    predict = predict.argmax(axis=0)

    return predict, only_infer_time

def func_ort_wrapped(model,frame):
    frame = np.expand_dims(frame,axis=0)
    
    only_run = time.time()
    predict = model(frame)[0][0]
    only_infer_time = time.time()-only_run

    return predict, only_infer_time

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    def __init__(self,engine_path,dtype=np.float32,wrapped=False):
        print(f"\nTRT Engine init...")
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # builder
        self.runtime = trt.Runtime(self.logger)
    
        # Load engine
        self.engine = self.load_engine(self.runtime, self.engine_path)

        # memory 할당
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(wrapped)
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self,wrapped):

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
            # ---- for cpu input ----
            if self.engine.binding_is_input(binding):
                # np.ndarray의 pagelocked를 할당
                host_mem = cuda.pagelocked_empty(size, self.dtype)
                # device memory 할당
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                inputs = HostDeviceMem(host_mem, device_mem)
            else:
                # np.ndarray의 pagelocked를 할당
                if wrapped:
                    host_mem = cuda.pagelocked_empty(size, np.int32)
                else:
                    host_mem = cuda.pagelocked_empty(size, np.float32)
                    
                # device memory 할당
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                outputs = HostDeviceMem(host_mem, device_mem)
        
            bindings.append(int(device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self,inputs):
        inputs = inputs.astype(self.dtype)
        inputs = np.ascontiguousarray(inputs)

        cuda.memcpy_htod_async(self.inputs.device, inputs, self.stream)
        # infer time check
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self.outputs.host, self.outputs.device, self.stream)
        self.stream.synchronize()

        return self.outputs.host

    def __del__(self):
        if self.engine is not None:
            del self.engine

class ortModel:
    def __init__(self,onnx_path,input_dtype=np.float32):
        
        self.in_dtype = input_dtype

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        device = onnxruntime.get_device()
        if device =="GPU":
            self.session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            self.session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        self.output_type = self.session.get_outputs()[0].type
        
        self.binding = self.set_io_binding()
    
    def inout_print(self):
        print(f"input name : {self.input_name}")
        print(f"input shape : {self.input_shape}")
        print(f"input type : {self.input_type}")
        
        print(f"output name : {self.output_name}")
        print(f"output shape : {self.output_shape}")
        print(f"output type : {self.output_type}")
        
    def set_io_binding(self):
        self.inout_print()
        binding = self.session.io_binding()
        binding.bind_cpu_input('inputs',np.empty(self.input_shape))
        binding.bind_output('outputs')
        return binding
        
    def __call__(self,inputs):
        
        inputs = inputs.astype(self.in_dtype)
        self.binding.bind_cpu_input('inputs',inputs)
        
        self.session.run_with_iobinding(self.binding)
        
        return self.binding.copy_outputs_to_cpu()[0]
        
def lib_version():
    # os.environ['CUDA_LAUNCH_BLOCKING']='1'
    print(f"\ntrt version : {trt.__version__}")
    print(f"torch version : {torch.__version__}")
    print(f"onnx verison : {onnx.__version__}")
    print(f"onnxruntime version : {onnxruntime.__version__}")

if __name__=='__main__':
    kargs = vars(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = kargs['checkpoint']

    # version check
    # lib_version() 
    if os.path.exists(model_path):
        if kargs['torch']:
            # check torch model
            if '.pth' in model_path:
                print(f"{model_path} model loading ...")
                model = torch.load(model_path)
                if kargs["wrapped"]:
                    model = WrappedModel(model).to(device)
                model.eval()
            else:
                print(f"{model_path} is not pytorch checkpoint")
                exit(1)
        elif kargs['trt']:
            if model_path.endswith(".engine"):
                TRT_LOGGER = trt.Logger()
                print(f"{model_path} tensorrt engine loading ...")
                model = TrtModel(model_path,wrapped=kargs['wrapped'])
            else:
                print(f"{model_path} is not tensorrt engine")
                exit(1)
        elif kargs['ort']:
            if model_path.endswith(".onnx"):
                print(f"{model_path} model loading ...")
                model = ortModel(onnx_path=model_path)
            else:
                print(f"{model_path} is not onnx model")
                exit(1)
                
        else:
            print("select option --torch or --trt or --ort")
            exit(1)
    else:
        print(f"{model_path} is not exist")
        exit(1)

    print("Model Loading Done.")
    
    # Inference function load
    if kargs['torch']:
        if kargs['wrapped']:
            func_inference = func_torch_wrapped
        else:
            func_inference = func_torch_plain
    elif kargs['trt']:
        if kargs['wrapped']:
            func_inference = func_trt_wrapped
        else:
            func_inference = func_trt_plain
    elif kargs['ort']:
        if kargs['wrapped']:
            func_inference = func_ort_wrapped
        else:
            func_inference = func_ort_plain
    
    # cmap load
    if kargs['num_classes'] == 21 or 'voc2012' in kargs['checkpoint']:
        cmap = cmap_voc
    elif kargs['num_classes'] == 19 or 'cityscapes' in kargs['checkpoint']:
        cmap = cmap_cityscapes
    cmap = np.array(cmap,dtype=np.uint8)

    # --------------------------------------------
    # video info check
    # --------------------------------------------
    
    out_name = os.path.basename(kargs['checkpoint']).split('.')[0] + '.mp4'
    if kargs['cam']:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera open failed")
            exit(1)
    else:
        if not os.path.exists(kargs['video']):
            print('input video does not exist\n')
            exit(1)
        cap = cv2.VideoCapture(kargs['video'])
        out_name = kargs['video'][:-4] + '_' + out_name

    frame_width = 640
    frame_height = 360
    
    if kargs['cam']:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    # cap.set(cv2.CAP_PROP_FPS,4)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'frame size ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))},{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}), {fps} fps')

    # ----------------------------------------------
    # video write
    # ----------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_cap = cv2.VideoWriter(os.path.join('result',out_name),fourcc,fps,(frame_width,frame_height))
    print(f"{out_name} encoding ...")

    total_frame = 0
    only_infer_time = 0
    num_frames = 150
    
    if kargs['torch']:
        print("Running pytorch...\n")
    elif kargs['trt']:
        print("TRT Engine running...\n")
    else:
        print("Onnx runtime running...\n")
    with torch.no_grad():
        start = time.time()
        while total_frame < num_frames:
            ret, org_frame = cap.read()
            if not ret:
                print('cap.read is failed')
                break
            total_frame +=1
            
            if not kargs['cam']:
                org_frame = cv2.resize(org_frame, (frame_width,frame_height))

            frame = org_frame.copy()
            # Inference
            predict, t = func_inference(model,frame)
            only_infer_time += t
            # colorize and write frame
            img = mask_colorize(predict,cmap)
            img = cv2.addWeighted(img,0.3,org_frame,0.7,0)
            
            out_cap.write(img)
        total_time = time.time()-start
    del(model)
    print(f'finish encoding - {out_name}')
    print(f'total frame = {total_frame} \ntotal time = {total_time:.2f}s')
    if total_frame:
        print(f'average time = {total_time/total_frame:.2f}s')
    print(f'Only inference time : {only_infer_time:.2f}s')
    
