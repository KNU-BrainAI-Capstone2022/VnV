# torch == 1.9.0
# onnx == 1.8.1
# onnx.ai.opeset == 13
# onnxruntime == 1.7.0 

import os
import torch
import argparse
from trt_model import *

def get_args():
    parser = argparse.ArgumentParser(description="Convert Pytorch Model to Onnx and Tensorrt Model")
    parser.add_argument("checkpoint", type=str, help='pytorch model weight path')
    parser.add_argument('-i',"--img_size", nargs='+', type=int, default=[640, 360], help="input image size (width height)")
    parser.add_argument('-O',"--output", type=str, default=None, help='Output model name (without extension)')
    parser.add_argument('-d',"--device", type=str, default=None, help='Device')
    
    # tensorrt engine from trtexec options
    parser.add_argument("--trtexec", action='store_true', help='Create trt engine(fp16) using trtexec')
    parser.add_argument("--onnx-opset", type=int, default=13, help='Opset version ai.onnx')
    
    # tensorrt engine from onnxParser options
    parser.add_argument("--onnxparser", action='store_true', help='Create trt engine(fp16) using onnxparser')
    parser.add_argument("--max_batch_size", type=int, default=1, help='explicit max_batch_size in tensorrt builder (default: 1)')
    parser.add_argument("--max_workspace", type=int, default=31, help='max_workscpace for tensorrt build (default: 2GB)')
    
    # Wrapped Model options
    parser.add_argument("--wrapped", action='store_true', help='Using wrapped model')

    return parser.parse_args()
    
if __name__=='__main__':
    kargs = vars(get_args())
    print(f'args : {kargs}')
    if not (kargs['trtexec'] or kargs['onnxparser']): # any converting flag, exit
        print("Select --trxexec or --onnxparser convert option")
        exit(1)
    if kargs['device'] == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device : {device}')

    if kargs['output'] is None:
        output_name = kargs['checkpoint'][:-4]
    else:
        output_name = kargs['output']
        
    # input image size
    w,h = kargs['img_size']
    
    if kargs['wrapped']: # wrapped model
        output_name = output_name + '_wrapped'
        input_shape = (1,h,w,3)
    else: # plain model
        output_name = output_name + '_plain'
        input_shape = (1,3,h,w)
        
    onnx_name = output_name + '.onnx'
    
    # create onnx model
    if not os.path.exists(onnx_name):
        # load torch model
        print(f'Load model....')
        model = torch.load(kargs['checkpoint'])

        # if model output is dictionary, convert nondict output model.
        if 'voc2012' in kargs['checkpoint']:
            model = Model_nondict(model)
            
        # convert to wrapped model
        if kargs['wrapped']:
            model = WrappedModel(model)
        
        model.eval()
        model = model.cuda()

        # input is Cuda Tensor and dtype float32.
        input_size = torch.randn(input_shape,dtype=torch.float32).cuda()
        
        print(f'input shape : {input_size.shape} ({input_size.dtype})')

    # torch --> onnx
    if kargs['onnxparser'] or kargs['trtexec']:
        if not os.path.exists(onnx_name):
            print(f'\nCreating onnx file...')
            torch.onnx.export(
                model,                      # 모델
                input_size,                 # 모델 입력값
                onnx_name,                  # 모델 저장 경로
                verbose=True,               # 변환 과정
                export_params=True,         # 모델 파일 안에 학습된 모델 가중치 저장
                do_constant_folding=True,   # 최적화시 상수 폴딩
                opset_version = kargs['onnx_opset'],         # onnx 버전
                input_names=['inputs'],      # 모델의 입력값을 가리키는 이름
                output_names= ['outputs'],   # 모델의 아웃풋 이름
                operator_export_type = torch.onnx.OperatorExportTypes.ONNX
            )
            print(f"{onnx_name} -> onnx is done")
            print("Please try again same command ")
            exit(1)

    if kargs['trtexec']:
        # onnx - > tensorrt
        engine_name = output_name + '_trtexec.engine'
        print(f"\n onnx -> tensorrt converting ...")
        os.system(f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_name} --saveEngine={engine_name} --fp16 --verbose --buildOnly")
        
    if kargs['onnxparser']:
        import tensorrt as trt
        
        
        engine_name = output_name + '_onnxparser.engine'

        logger = trt.Logger(trt.Logger.INFO)
        with trt.Builder(logger) as builder:
            builder.max_batch_size = kargs['max_batch_size']
            network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with builder.create_network(network_flag) as net:
                with trt.OnnxParser(net, logger) as p:
                    with open(onnx_name, 'rb') as f:
                        if not p.parse(f.read()):
                            for err in range(p.num_errors):
                                print(p.get_error(err))
                        else:
                            logger.log(trt.Logger.INFO, 'Onnx file parsed successfully')

                config=builder.create_builder_config()

                config.max_workspace_size = 1 << kargs['max_workspace']

                config.set_flag(trt.BuilderFlag.FP16)
        
                config.default_device_type = trt.DeviceType.GPU

                config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE

                logger.log(trt.Logger.INFO, 'Checking if network is supported...')

                if builder.is_network_supported(net, config):
                    logger.log(trt.Logger.INFO, 'Network is supported')
                else:
                    logger.log(trt.Logger.ERROR, 'Network contains operations that are not supported by TensorRT')
                    logger.log(trt.Logger.ERROR, 'QUITTING because network is not supported')
                    quit()

                logger.log(trt.Logger.INFO, 'Building inference engine...')
                engine = builder.build_engine(net, config)

                if engine != None:
                    logger.log(trt.Logger.INFO, 'Inference engine built successfully')
                else:
                    logger.log(trt.Logger.ERROR,'Inference engine built failed')
                    quit()

                with open(engine_name, 'wb') as s:
                    s.write(engine.serialize())
                logger.log(trt.Logger.INFO, f'Inference engine saved to {engine_name}')