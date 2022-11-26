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
    def __init__(self,engine_path,dtype=np.float16):
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

        # print(f"engine.get_location -> {self.engine.get_location(0)}")

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
                # print(f"input data host_mem : {host_mem}, device_mem :{device_mem}")
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream

    def __call__(self,x,batch_size=1):
        
        # -------------------------- torch tensor
        x = x.half().contiguous()
        # self.inputs[0].host = x.ravel()

        # --------------------------- numpy
        # x = x.astype(self.dtype)
        # x = np.ascontiguousarray(x)
        
        # # x.ravel is Returns to a continuous 1-dimensional plane
        np.copyto(self.inputs[0].host,x.ravel())
        # ----------------------------------------------

        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # infer time check
        only_run = time.time()
        check = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        i_time = time.time()-only_run

        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

        self.stream.synchronize()

        return [out.host.reshape(batch_size,-1) for out in self.outputs], i_time

    def __del__(self):
        if self.engine is not None:
            del self.engine

class Load_engine:
    def __init__(
            self,
            max_batch_size=1,
            onnx_path=None,
            maxworkspace = 25,
            precision_str= "FP16",
            precision=None,
            allowGPUFallback=None,
            dla_core=None,
            device=None,
            calibrator=None,
            engine_path = None
            ):

        self.max_batch_size = max_batch_size
        self.onnx_path = onnx_path
        self.maxworkspace = maxworkspace
        self.precision_str = precision_str
        self.precision = precision
        self.allowGPUFallback = allowGPUFallback
        self.dla_core = dla_core
        self.device = device
        self.calibrator = calibrator
        if engine_path == None:
            self.engine_path = onnx_path.replace(".onnx", ".engine")
        else:
            self.engine_path = engine_path


    def parse_or_load(self):
        logger = trt.Logger(trt.Logger.INFO)

        with trt.Builder(logger) as builder:
            builder.max_batch_size=self.max_batch_size
            #setting max_batch_size isn't strictly necessary in this case
            #since the onnx file already has that info, but its a good practice
            
            network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            
            #since the onnx file was exported with an explicit batch dim,
            #we need to tell this to the builder. We do that with EXPLICIT_BATCH flag
            
            with builder.create_network(network_flag) as net:
                with trt.OnnxParser(net, logger) as p:
                    #create onnx parser which will read onnx file and
                    #populate the network object `net`          
                    with open(self.onnx_path, 'rb') as f:
                        if not p.parse(f.read()):
                            for err in range(p.num_errors):
                                print(p.get_error(err))
                        else:
                            logger.log(trt.Logger.INFO, 'Onnx file parsed successfully')
        
                    net.get_input(0).dtype=trt.DataType.HALF
                    net.get_output(0).dtype=trt.DataType.HALF
                    #we set the inputs and outputs to be float16 type to enable
                    #maximum fp16 acceleration. Also helps for int8
                    
                    config=builder.create_builder_config()
                    #we specify all the important parameters like precision, 
                    #device type, fallback in config object
        
                    config.max_workspace_size = self.maxworkspace
        
                    if self.precision_str in ['FP16', 'INT8']:
                        config.flags = ((1<<self.precision)|(1<<self.allowGPUFallback))
                        config.DLA_core=self.dla_core
                    # DLA core (0 or 1 for Jetson AGX/NX/Orin) to be used must be 
                    # specified at engine build time. An engine built for DLA0 will 
                    # not work on DLA1. As such, to use two DLA engines simultaneously, 
                    # we must build two different engines.
        
                    config.default_device_type=self.device
                    #if device is set to GPU, DLA_core has no effect
        
                    config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE
                    #building with verbose profiling helps debug the engine if there are
                    #errors in inference output. Does not impact throughput.
        
                    if self.precision_str=='INT8' and self.calibrator is None:
                        logger.log(trt.Logger.ERROR, 'Please provide calibrator')
                        #can't proceed without a calibrator
                        quit()
                    elif self.precision_str=='INT8' and self.calibrator is not None:
                        config.int8_calibrator=self.calibrator
                        logger.log(trt.Logger.INFO, 'Using INT8 calibrator provided by user')
        
                    logger.log(trt.Logger.INFO, 'Checking if network is supported...')
                    
                    if builder.is_network_supported(net, config):
                        logger.log(trt.Logger.INFO, 'Network is supported')
                    #tensorRT engine can be built only if all ops in network are supported.
                    #If ops are not supported, build will fail. In this case, consider using 
                    #torch-tensorrt integration. We might do a blog post on this in the future.
                    else:
                        logger.log(trt.Logger.ERROR, 'Network contains operations that are not supported by TensorRT')
                        logger.log(trt.Logger.ERROR, 'QUITTING because network is not supported')
                        quit()
        
                    if self.device==trt.DeviceType.DLA:
                        dla_supported=0
                        logger.log(trt.Logger.INFO, 'Number of layers in network: {}'.format(net.num_layers))
                        for idx in range(net.num_layers):
                            if config.can_run_on_DLA(net.get_layer(idx)):
                                dla_supported+=1
        
                        logger.log(trt.Logger.INFO, f'{dla_supported} of {net.num_layers} layers are supported on DLA')
        
                    logger.log(trt.Logger.INFO, 'Building inference engine...')
                    engine=builder.build_engine(net, config)
                    #this will take some time
        
                    logger.log(trt.Logger.INFO, 'Inference engine built successfully')
        
                    with open(self.enginepath, 'wb') as s:
                        s.write(engine.serialize())
                    logger.log(trt.Logger.INFO, f'Inference engine saved to {self.enginepath}')
                
        return engine, logger

def lib_version():
    print(f"\ntrt version : {trt.__version__}")
    print(f"torch version : {torch.__version__}")
    print(f"onnx verison : {onnx.__version__}")
    print(f"onnxruntime version : {onnxruntime.__version__}")


if __name__=='__main__':
    kargs = vars(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kargs['torch']:
        model = deeplabv3plus_mobilenet["deeplabv3plus_mobilenet"](num_classes=19,output_stride=16,pretrained_backbone=False).to(device)
        model_path = kargs['base']
        model.load_state_dict(torch.load(model_path))
        model.eval()
    elif kargs['torch2trt']:
        model = TRTModule()
        model_path = kargs['base'].replace(".pth","_jetson_trt_fp16.pth")
        model.load_state_dict(torch.load(model_path))
    else:
        TRT_LOGGER = trt.Logger()
        engine_path = kargs['engine']
    # cmap load
    cmap = CustomCityscapesSegmentation.cmap

    # version print()
    lib_version()
    
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
            while total_frame <30:
                ret, frame = cap.read()
                if not ret:
                    print('cap.read is failed')
                    break
                frame = cv2.resize(frame, (frame_width,frame_height))
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                input_image = F.to_tensor(frame).unsqueeze(0).to(device, dtype=torch.float16)

                only_run = time.time()
                predict = model(input_image)
                only_infer_time += time.time()-only_run

                predict = predict.detach().squeeze(0).argmax(dim=0).cpu().numpy()
                predict = mask_colorize(predict,cmap).astype(np.uint8)
                
                result = cv2.addWeighted(frame,0.3,predict,0.7,0)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                
                out_cap.write(result)
                total_frame +=1
    else:
        print("TRT Engine running...\n")
        model = TrtModel(engine_path)
        start = time.time()
        total_frame =0
        only_infer_time = 0
        # read video
        while total_frame < 60:
            ret, frame = cap.read()
            if not ret:
                print('cap.read is failed')
                break
            total_frame +=1
            print(f'total_frame {total_frame}')
            frame = cv2.resize(frame,(frame_width,frame_height))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # input_image = preprocess(frame)
            input_image = F.to_tensor(frame)
            outputs,t = model(input_image)
            # print(len(output[0]))
            
            only_infer_time +=t
            # img = torch.argmax(torch.from_numpy(outputs[0]).view((19,frame_height,frame_width)),dim=0)
            # img = np.array(mask_colorize(img,cmap))

            img = np.argmax(np.reshape(outputs[0],(19,frame_height,frame_width)),axis=0)
            img = mask_colorize(img,cmap).astype(np.uint8)
            img = cv2.addWeighted(frame,0.3,img,0.7,0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            out_cap.write(img)
            # cv2.imwrite('../video/test.jpg',img)


    print(f'finish encoding - {out_name}')
    total_time = time.time()-start
    print(f'total frame = {total_frame} \ntotal time = {total_time:.2f}s')
    print(f'average time = {total_time/total_frame:.2f}s')
    print(f'Only inference time : {only_infer_time:.2f}s')
    