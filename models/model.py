from .utils import IntermediateLayerGetter, _SimpleSegmentationModel
from .deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet
from .fcn import FCN,FCNHead,FCN8

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride == 8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else: # 16
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _fcn_resnet(name, backbone_name, num_classes,pretrained_backbone):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone
    )
    inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer2':'layer2','layer3': 'layer3', 'layer4': 'layer4'}
    classifier = FCNHead(inplanes,num_classes)
    aux_classifier = FCNHead(aux_inplanes,num_classes)
    
    backbone = IntermediateLayerGetter(backbone,return_layers=return_layers)

    #model = FCN(backbone,classifier,aux_classifier)

    classifier = FCN8(inplanes,num_classes)
    model = _SimpleSegmentationModel(backbone, classifier)

    return model

def _load_model(arch_type, backbone, num_classes, output_stride=8, pretrained_backbone=True):
    if arch_type =='fcn':
        if backbone.startswith('resnet'):
            model = _fcn_resnet(arch_type, backbone,num_classes,pretrained_backbone)
        else:
            raise NotImplementedError

    else:
        if backbone.startswith('resnet'):
            model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
        else:
            raise NotImplementedError
    return model

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def fcn_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a FCN model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('fcn','resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def fcn_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a FCN model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('fcn','resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)