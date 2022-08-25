import argparse
import sys
import time
import warnings

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import attempt_load, End2End
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size, colorstr, check_version, file_size, check_requirements, check_dataset
from utils.torch_utils import select_device
from utils.add_nms import RegisterNMS

import onnx_setting

from models.common import has_Focus_layer

def reformat_img_wFocus(img, has_Focus:bool):
    if onnx_setting.export_onnx == True:
        import copy
        shape = img.shape
        img_out = copy.deepcopy(img)
        if has_Focus:
            img_out = img_out.view( shape[0], shape[1] * 4, shape[2]//2, shape[3]//2 )
    return img_out

def export_saved_model(model,
                       im,
                       file,
                       dynamic,
                       tf_nms=False,
                       agnostic_nms=False,
                       topk_per_class=100,
                       topk_all=100,
                       iou_thres=0.45,
                       conf_thres=0.25,
                       keras=False,
                       prefix=colorstr('TensorFlow SavedModel:')):
    # YOLOv5 TensorFlow SavedModel export
    # try:
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    # 
    onnx_setting.export_onnx = True
    im_reform = reformat_img_wFocus(im, has_Focus_layer(model))
    #
    from models.tf import TFDetect, TFModel

    print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    f = str(file).replace('.pt', '_saved_model')
    batch_size, ch, *imgsz = list(im_reform.shape)  # BCHW

    tf_model = TFModel(cfg=model.yaml, ch=ch, model=model, nc=model.nc, imgsz=imgsz)
    im_reform = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
    _ = tf_model.predict(im_reform, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    if keras:
        keras_model.save(f, save_format='tf')
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x)[0], [spec])
        tfm.__call__(im_reform)
        tf.saved_model.save(tfm,
                            f,
                            options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
                            if check_version(tf.__version__, '2.6') else tf.saved_model.SaveOptions())
    print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    return keras_model, f
    # except Exception as e:
    #     print(f'\n{prefix} export failure: {e}')
    #     return None, None


def export_tflite(keras_model, im, file, int8, data, nms=None, agnostic_nms=None, prefix=colorstr('TensorFlow Lite:'), has_Focus_layer=False, max_int8_img_cnt=100):
    # YOLOv5 TensorFlow Lite export
    # try:
    from utils.datasets import LoadImages
    import tensorflow as tf
    from yolov5_quant_utils import datasetGenerateImagesYolov5
    import yaml

    print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW
    # f = str(file).replace('.pt', '-fp16.tflite')
    f = str(file).replace('.pt', '-fp32.tflite')

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float32]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        from models.tf import representative_dataset_gen
        with open(data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        if has_Focus_layer:
            converter.representative_dataset = lambda: datasetGenerateImagesYolov5(image_size=imgsz, image_mask=data_dict['val']+'/*.jpg', maximum_match=max_int8_img_cnt, print_filenames=True )
        else:
            dataset = LoadImages(data_dict['val'], img_size=imgsz)
            converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=max_int8_img_cnt)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_quantizer = True
        converter._experimental_disable_per_channel=True    # disbable per channel quant
        f = str(file).replace('.pt', '-int8.tflite')
    if nms or agnostic_nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_model = converter.convert()
    open(f, "wb").write(tflite_model)
    print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    return f
    # except Exception as e:
    #     print(f'\n{prefix} export failure: {e}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    parser.add_argument('--max-int8-img-cnt', type=int, default=100, help='max num images used for quantization')
    parser.add_argument('--tflite', action='store_true', help='export tflite')
    parser.add_argument('--onnx', action='store_true', help='export onnx')
    parser.add_argument('--torchscript', action='store_true', help='export torchscript')
    
    
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    
    onnx_setting.export_onnx = False
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    # TorchScript export
    if opt.torchscript:
        try:
            print('\nStarting TorchScript export with torch %s...' % torch.__version__)
            f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
            ts = torch.jit.trace(model, img, strict=False)
            ts.save(f)
            print('TorchScript export success, saved as %s' % f)
        except Exception as e:
            print('TorchScript export failure: %s' % e)

        # CoreML export
        try:
            import coremltools as ct

            print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
            # convert model from torchscript and apply pixel scaling as per detect.py
            ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
            bits, mode = (8, 'kmeans_lut') if opt.int8 else (16, 'linear') if opt.fp16 else (32, None)
            if bits < 32:
                if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                        ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
                else:
                    print('quantization only supported on macOS, skipping...')

            f = opt.weights.replace('.pt', '.mlmodel')  # filename
            ct_model.save(f)
            print('CoreML export success, saved as %s' % f)
        except Exception as e:
            print('CoreML export failure: %s' % e)
                        
        # TorchScript-Lite export
        try:
            print('\nStarting TorchScript-Lite export with torch %s...' % torch.__version__)
            f = opt.weights.replace('.pt', '.torchscript.ptl')  # filename
            tsl = torch.jit.trace(model, img, strict=False)
            tsl = optimize_for_mobile(tsl)
            tsl._save_for_lite_interpreter(f)
            print('TorchScript-Lite export success, saved as %s' % f)
        except Exception as e:
            print('TorchScript-Lite export failure: %s' % e)

    # ONNX export
    if opt.onnx:
        try:
            import onnx

            print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
            f = opt.weights.replace('.pt', '.onnx')  # filename
            model.eval()
            output_names = ['classes', 'boxes'] if y is None else ['output']
            dynamic_axes = None
            if opt.dynamic:
                dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                'output': {0: 'batch', 2: 'y', 3: 'x'}}
            if opt.dynamic_batch:
                opt.batch_size = 'batch'
                dynamic_axes = {
                    'images': {
                        0: 'batch',
                    }, }
                if opt.end2end and opt.max_wh is None:
                    output_axes = {
                        'num_dets': {0: 'batch'},
                        'det_boxes': {0: 'batch'},
                        'det_scores': {0: 'batch'},
                        'det_classes': {0: 'batch'},
                    }
                else:
                    output_axes = {
                        'output': {0: 'batch'},
                    }
                dynamic_axes.update(output_axes)
            if opt.grid:
                if opt.end2end:
                    print('\nStarting export end2end onnx model for %s...' % 'TensorRT' if opt.max_wh is None else 'onnxruntime')
                    model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device)
                    if opt.end2end and opt.max_wh is None:
                        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                        shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                                opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                    else:
                        output_names = ['output']
                else:
                    model.model[-1].concat = True

            onnx_setting.export_onnx = True
            im_reform = reformat_img_wFocus(img, has_Focus_layer(model))

            torch.onnx.export(model, im_reform, f, verbose=False, opset_version=12, input_names=['images'],
                            output_names=output_names,
                            dynamic_axes=dynamic_axes)

            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model

            if opt.end2end and opt.max_wh is None:
                for i in onnx_model.graph.output:
                    for j in i.type.tensor_type.shape.dim:
                        j.dim_param = str(shapes.pop(0))

            # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

            # # Metadata
            # d = {'stride': int(max(model.stride))}
            # for k, v in d.items():
            #     meta = onnx_model.metadata_props.add()
            #     meta.key, meta.value = k, str(v)
            # onnx.save(onnx_model, f)

            if opt.simplify:
                try:
                    import onnxsim

                    print('\nStarting to simplify ONNX...')
                    onnx_model, check = onnxsim.simplify(onnx_model)
                    assert check, 'assert check failed'
                except Exception as e:
                    print(f'Simplifier failure: {e}')

            # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
            onnx.save(onnx_model,f)
            print('ONNX export success, saved as %s' % f)

            if opt.include_nms:
                print('Registering NMS plugin for ONNX...')
                mo = RegisterNMS(f)
                mo.register_nms()
                mo.save(f)

        except Exception as e:
            print('ONNX export failure: %s' % e)


    # TensorFlow Exports
    if opt.tflite:
        if opt.int8:  # TFLite --int8 bug https://github.com/ultralytics/yolov5/issues/5707
            check_requirements(('flatbuffers==1.12',))  # required before `import tensorflow`
        keras_model, _ = export_saved_model(model.cpu(),
                                         img,
                                         file=opt.weights,
                                         dynamic=False,
                                        )
        _ = export_tflite(keras_model, img, file=opt.weights, int8=opt.int8, data=opt.data if opt.int8 else None, has_Focus_layer=has_Focus_layer(model), max_int8_img_cnt=opt.max_int8_img_cnt)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
