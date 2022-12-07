# torch == 1.9.0
# onnx == 1.8.1
# onnx.ai.opeset == 13
# onnxruntime == 1.7.0 

import sys
import os
sys.path.append("../")
import torch
import argparse
from models.model import deeplabv3plus_resnet50,deeplabv3plus_mobilenet

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument('-c',"--checkpoint", type=str, default='./checkpoint/mobilenet_plain.pth', help='pytorch weight path')
    parser.add_argument('-i',"--img_size", nargs='+', type=int, default=[360, 640], help="input image size (height width)")
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: 19, cityscapes)")
    parser.add_argument('-O',"--output", type=str, default=None, help='output model name')
    
    # tensorrt engine from trtexec options
    parser.add_argument("--trtexec", action='store_true', help='Create trt engine(fp16) and onnx model using trtexec')
    parser.add_argument("--onnx-opset", type=int, default=13, help='Opset version ai.onnx')
    
    # tensorrt engine from onnxParser options
    parser.add_argument("--onnxparser", action='store_true', help='Create trt engine(fp16) and onnx model using onnxparser')
    
    # torch2trt options
    parser.add_argument("--torch2trt", action='store_true', help='Create tensorrt model using torch2trt')
    parser.add_argument("--int8", action='store_true', help="Create torch2trt int8")
    parser.add_argument("--fp16", action='store_true', help="Create torch2trt fp16")
    parser.add_argument("--fp32", action='store_true', help="Create torch2trt fp32")
    
    # Wrapped Model options
    parser.add_argument("--wrapped", action='store_true', help='Using wrapped model(v2)')
    
    kargs = vars(parser.parse_args())
    print(f'args : {kargs}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device : {device}')

    if kargs['output'] is None:
        output_name = kargs['checkpoint'][:-4]
    else:
        output_name = kargs['output']
        
    onnx_name = output_name + '.onnx'
    if not os.path.exists(onnx_name) or kargs['torch2trt']:
        if 'resnet50' in kargs['checkpoint']:
            model = deeplabv3plus_resnet50(num_classes=kargs['num_classes'],pretrained_backbone=False)
        elif 'mobilenet' in kargs['checkpoint']:
            model = deeplabv3plus_mobilenet(num_classes=kargs['num_classes'], pretrained_backbone=False)
        

        # load weight
        print(f'Load model....')
        model_state= torch.load(kargs['checkpoint'])
        model.load_state_dict(model_state['model_state'])

        # input image size
        h,w = kargs['img_size']
        input_shape = (1,3,h,w)
        
        if kargs['wrapped']:
            from wrapmodel import WrappedModel
            output_name = output_name.replace('plain','wrapped')
            input_shape = (1,h,w,3)
            model = WrappedModel(model)
        
        model.eval()
        model = model.cuda()

        # input is Cuda Tensor
        input_size = torch.randn(input_shape,dtype=torch.float32).cuda()
        
        print(f'input shape : {input_size.shape} ({input_size.dtype})')

    # torch --> onnx
    if kargs['trtexec'] or kargs['onnxparser']:
        if not os.path.exists(onnx_name):
            print(f'\nCreating onnx file...')
            torch.onnx.export(
                model,                      # 모델
                input_size,                 # 모델 입력값
                onnx_name,                  # 모델 저장 경로
                verbose=True,              # 변환 과정
                export_params=True,         # 모델 파일 안에 학습된 모델 가중치 저장
                opset_version = kargs['onnx_opset'],         # onnx 버전
                input_names=['inputs'],      # 모델의 입력값을 가리키는 이름
                output_names= ['outputs'],   # 모델의 아웃풋 이름
                operator_export_type = torch.onnx.OperatorExportTypes.ONNX
            )
            print(f"{onnx_name} -> onnx is done")

    if kargs['trtexec']:
        # onnx - > tensorrt
        engine_name = output_name + '_trtexec.engine'
        print(f"\n onnx -> tensorrt converting ...")
        os.system(f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_name} --saveEngine={engine_name} --fp16 --verbose --buildOnly")
        # /usr/src/tensorrt/bin/trtexec --onnx=model_best_jetson.onnx --saveEngine=model_best_jetson_fp16.engine --fp16 --verbose --buildOnly
        
    if kargs['onnxparser']:
        import tensorrt as trt
        
        def onnx_parsing_trt(onnx_path, max_batch_size=1, max_workspace=30, output_name=None):
            engine_name = output_name + '_onnxparser.engine'

            logger = trt.Logger(trt.Logger.INFO)
            with trt.Builder(logger) as builder:
                builder.max_batch_size = max_batch_size
                network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                with builder.create_network(network_flag) as net:
                    with trt.OnnxParser(net, logger) as p:
                        with open(onnx_path, 'rb') as f:
                            if not p.parse(f.read()):
                                for err in range(p.num_errors):
                                    print(p.get_error(err))
                            else:
                                logger.log(trt.Logger.INFO, 'Onnx file parsed successfully')
                    # Topk Layer add
                    topk = net.add_topk(input=net.get_output(0), op=trt.tensorrt.TopKOperation.MAX, k = 1, axes=2)
                    topk.name = 'TopK_240'
                    topk.get_output(1).name = 'outputs_topk'
                    
                    net.unmark_output(net.get_output(0))
                    net.mark_output(topk.get_output(1))
                    
                    config=builder.create_builder_config()

                    config.max_workspace_size = 1 << max_workspace

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
            return
        onnx_parsing_trt(onnx_path=onnx_name, output_name=output_name)
            
    # torch -> tensorrt 
    if kargs['torch2trt']:
        from torch2trt import torch2trt
        if kargs['int8']:
            mode = 'int8'
        elif kargs['fp16']:
            mode = 'fp16'
        else:
            mode = 'fp32'
        mode_kargs = {'int8':{'int8_mode':True},'fp16':{'fp16_mode':True},'fp32':{'int32_mode':True}}

        engine_name = f"{output_name}_torch2trt_{mode}.pth"
        print(f'\nCreating trt {mode} file...')
        trt_model = torch2trt(model,[input_size], max_workspace_size=1 << 32,**mode_kargs[mode],use_onnx=True,onnx_opset=kargs['onnx_opset'])

        torch.save(trt_model.state_dict(),engine_name)
        print(f"\nTRTModule {engine_name} is Created")
