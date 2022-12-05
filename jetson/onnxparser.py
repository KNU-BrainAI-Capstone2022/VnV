# torch == 1.9.0
# onnx == 1.8.1
# onnx.ai.opeset == 13
# onnxruntime == 1.7.0 

import argparse
import tensorrt as trt

class Load_engine:
    def __init__(self,
                max_batch_size=1,
                onnx_path=None,
                maxworkspace = 32,
                engine_path = None
                ):

        self.max_batch_size = max_batch_size
        self.onnx_path = onnx_path
        self.maxworkspace = maxworkspace
        if engine_path == None:
            self.engine_path = onnx_path.replace(".onnx", "_onnxparser.engine")
        else:
            self.engine_path = engine_path


    def parse_or_load(self):
        logger = trt.Logger(trt.Logger.INFO)

        # with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        #     new_engine = runtime.deserialize_cuda_engine(f.read())
        # inspect = new_engine.create_engine_inspector()
        # print(inspect.get_engine_information(trt.tensorrt.LayerInformationFormat.ONELINE))
        # for i in range(new_engine.num_layers):
        #     print(i, inspect.get_layer_information(i,trt.tensorrt.LayerInformationFormat.ONELINE))
        # quit()
        with trt.Builder(logger) as builder:
            builder.max_batch_size = self.max_batch_size
            #setting max_batch_size isn't strictly necessary in this case
            #since the onnx file already has that info, but its a good practice

            network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            #since the onnx file was exported with an explicit batch dim,
            #we need to tell this to the builder. We do that with EXPLICIT_BATCH flag

            with builder.create_network(network_flag) as net:
                with trt.OnnxParser(net, logger) as p:
                    with open(self.onnx_path, 'rb') as f:
                        if not p.parse(f.read()):
                            for err in range(p.num_errors):
                                print(p.get_error(err))
                        else:
                            logger.log(trt.Logger.INFO, 'Onnx file parsed successfully')

                # print("type(net) :", type(net))
                # net:trt.INetworkDefinition
                # print("net.num_inputs :", net.num_inputs)
                # print("net.num_outputs :", net.num_outputs)
                # print("net.num_layers :", net.num_layers)
                
                # print("net.get_output(0).name :", net.get_output(0).name)
                
                # topk = net.add_topk(input=net.get_output(0), op=trt.tensorrt.TopKOperation.MAX, k = 1, axes=2)
                # topk.name = 'TopK_240'
                # topk.get_output(0).name = 'outputs_topk'
                
                # net.unmark_output(net.get_output(0))
                # net.mark_output(topk.get_output(0))
                
                # print("net.get_output(0).name :", net.get_output(0).name)
                # print("net.get_output(0).shape :", net.get_output(0).shape)
                
                # print("net.num_inputs :", net.num_inputs)
                # print("net.num_outputs :", net.num_outputs)
                # print("net.num_layers :", net.num_layers)
                
                # net.get_input(0).dtype=trt.DataType.HALF
                # net.get_output(0).dtype=trt.DataType.HALF
                
                # inp = net.get_input(0)
                # out = net.get_output(0)
                
                # print("inp.dtype :", inp.dtype)
                # print("out.dtype :", out.dtype)
                
                # print("inp.shape :", inp.shape)
                # print("out.shape :", out.shape)
                
                # print("type(inp) :", type(inp))
                # print("type(out) :", type(out))
                
                # print("inp.location :", inp.location)
                # print("out.location :", out.location)
                
                # print("inp.is_network_input :", inp.is_network_input)
                # print("out.is_network_output :", out.is_network_output)
                
                #import time
                #time.sleep(5)
                #quit()
                # for i in range(net.num_layers):
                #     ilayer=net.get_layer(i)
                #     print(f"{i}'th layer's name, type : {ilayer.name}, {ilayer.type}")
                #     print(f"{i}'th layer's inp,out shape : {ilayer.num_inputs}, {ilayer.num_outputs}")
                #     print(f"{i}'th layer's precision : {ilayer.precision_is_set}, {ilayer.precision}")
                #     print()
                
                
                config=builder.create_builder_config()

                config.max_workspace_size = 1 << self.maxworkspace

                config.set_flag(trt.BuilderFlag.FP16)
                # config.flags = 1 << trt.BuilderFlag.FP16
                
                config.default_device_type = trt.DeviceType.GPU
                #if device is set to GPU, DLA_core has no effect

                config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE
                # building with verbose profiling helps debug the engine if there are
                # errors in inference output. Does not impact throughput.

                logger.log(trt.Logger.INFO, 'Checking if network is supported...')

                if builder.is_network_supported(net, config):
                    logger.log(trt.Logger.INFO, 'Network is supported')
                else:
                    logger.log(trt.Logger.ERROR, 'Network contains operations that are not supported by TensorRT')
                    logger.log(trt.Logger.ERROR, 'QUITTING because network is not supported')
                    quit()

                logger.log(trt.Logger.INFO, 'Building inference engine...')
                engine = builder.build_engine(net, config)
                # this will take some time
                if engine != None:
                    logger.log(trt.Logger.INFO, 'Inference engine built successfully')
                else:
                    logger.log(trt.Logger.ERROR,'Inference engine built failed')
                    quit()

                with open(self.engine_path, 'wb') as s:
                    s.write(engine.serialize())
                logger.log(trt.Logger.INFO, f'Inference engine saved to {self.engine_path}')
        return

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument('-c',"--onnx_path", type=str, default='./checkpoint/plain.pth', help='onnx weight path')
    parser.add_argument('-O',"--output", type=str, default=None, help='output model name')    
    return parser.parse_args()
    
if __name__=='__main__':
    kargs = vars(get_args())
    print(f'args : {kargs}')
    # onnx --> trt engine
    loader = Load_engine(onnx_path = kargs['onnx_path']).parse_or_load()
