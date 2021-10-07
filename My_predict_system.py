import os
import sys
import subprocess
import cv2
import copy
import numpy as np
import time
import math
import string
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
import pyclipper
import imghdr
import paddle

# 待优化。。。
__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
                                img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        else:
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch


def det_create_predictor(det_model_dir, use_gpu, gpu_mem=500, use_tensorrt=False,
                         use_fp16=False, max_batch_size=10, enable_mkldnn=False):
    # enable_mkldnn=False, rec_batch_num=6):
    # if mode == "det":
    #     model_dir = args.det_model_dir
    # elif mode == 'cls':
    #     model_dir = args.cls_model_dir
    # elif mode == 'rec':
    #     model_dir = args.rec_model_dir
    # else:
    #     model_dir = args.e2e_model_dir

    if det_model_dir is None:
        # logger.info("not find {} model file path {}".format(mode, model_dir))
        print("not find model file path {}".format(det_model_dir))
        sys.exit(0)
    model_file_path = det_model_dir + "/inference.pdmodel"
    params_file_path = det_model_dir + "/inference.pdiparams"
    if not os.path.exists(model_file_path):
        # logger.info("not find model file path {}".format(model_file_path))
        print("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        # logger.info("not find params file path {}".format(params_file_path))
        "not find params file path {}".format(params_file_path)
        sys.exit(0)

    config = paddle.inference.Config(model_file_path, params_file_path)

    if use_gpu:
        config.enable_use_gpu(gpu_mem, 0)
        if use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=paddle.inference.PrecisionType.Half
                if use_fp16 else paddle.inference.PrecisionType.Float32,
                max_batch_size=max_batch_size)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(6)
        if enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            #  TODO LDOUBLEV: fix mkldnn bug when bach_size  > 1
            # config.set_mkldnn_op({'conv2d', 'depthwise_conv2d', 'pool2d', 'batch_norm'})
            # rec_batch_num = 1

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.switch_use_feed_fetch_ops(False)

    # create predictor
    predictor = paddle.inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors
    # return predictor, input_tensor, output_tensors, rec_batch_num


class TextDetector(object):
    def __init__(self, model_dir="./inference/ch_ppocr_server/ch_ppocr_server_v2.0_det_infer", det_algorithm='DB',
                 use_gpu=True, enable_mkldnn=False, det_limit_side_len=960, det_limit_type='max', det_db_thresh=0.3,
                 det_db_box_thresh=0.5, det_db_unclip_ratio=1.6, use_dilation=False, det_db_score_mode='fast'):
        # self.args = args
        self.det_algorithm = det_algorithm
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': det_limit_side_len,
                'limit_type': det_limit_type
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = det_db_thresh
            postprocess_params["box_thresh"] = det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = det_db_unclip_ratio
            postprocess_params["use_dilation"] = use_dilation
            # if hasattr(args, "det_db_score_mode"):
            postprocess_params["score_mode"] = det_db_score_mode

        # elif self.det_algorithm == "EAST":
        #     postprocess_params['name'] = 'EASTPostProcess'
        #     postprocess_params["score_thresh"] = args.det_east_score_thresh
        #     postprocess_params["cover_thresh"] = args.det_east_cover_thresh
        #     postprocess_params["nms_thresh"] = args.det_east_nms_thresh
        # elif self.det_algorithm == "SAST":
        #     pre_process_list[0] = {
        #         'DetResizeForTest': {
        #             'resize_long': args.det_limit_side_len
        #         }
        #     }
        #     postprocess_params['name'] = 'SASTPostProcess'
        #     postprocess_params["score_thresh"] = args.det_sast_score_thresh
        #     postprocess_params["nms_thresh"] = args.det_sast_nms_thresh
        #     self.det_sast_polygon = args.det_sast_polygon
        #     if self.det_sast_polygon:
        #         postprocess_params["sample_pts_num"] = 6
        #         postprocess_params["expand_scale"] = 1.2
        #         postprocess_params["shrink_ratio_of_width"] = 0.2
        #     else:
        #         postprocess_params["sample_pts_num"] = 2
        #         postprocess_params["expand_scale"] = 1.0
        #         postprocess_params["shrink_ratio_of_width"] = 0.3
        else:
            # logger.info("unknown det_algorithm:{}".format(self.det_algorithm))
            print("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        # self.postprocess_op = build_post_process(postprocess_params)
        self.postprocess_op = eval(postprocess_params['name'])(**postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors = \
            det_create_predictor(det_model_dir=self.model_dir, use_gpu=self.use_gpu,
                                 enable_mkldnn=self.enable_mkldnn)
        # self.predictor, self.input_tensor, self.output_tensors = utility.create_predictor(
        #     args, 'det', logger)  # paddle.jit.load(args.det_model_dir)
        # self.predictor.eval()

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        # if self.det_algorithm == "EAST":
        #     preds['f_geo'] = outputs[0]
        #     preds['f_score'] = outputs[1]
        # elif self.det_algorithm == 'SAST':
        #     preds['f_border'] = outputs[0]
        #     preds['f_score'] = outputs[1]
        #     preds['f_tco'] = outputs[2]
        #     preds['f_tvo'] = outputs[3]
        # elif self.det_algorithm == 'DB':
        if self.det_algorithm == 'DB':
            preds['maps'] = outputs[0]
        else:
            raise NotImplementedError
        self.predictor.try_shrink_memory()
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        # if self.det_algorithm == "SAST" and self.det_sast_polygon:
        #     dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        # else:
        #     dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        support_character_type = [
            'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
            'it', 'xi', 'pu', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs', 'oc',
            'rsc', 'bg', 'uk', 'be', 'te', 'ka', 'chinese_cht', 'hi', 'mr',
            'ne', 'EN', 'latin', 'arabic', 'cyrillic', 'devanagari'
        ]
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, character_type)

        self.beg_str = "sos"
        self.end_str = "eos"

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "EN_symbol":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        elif character_type in support_character_type:
            self.character_str = ""
            assert character_dict_path is not None, \
                "character_dict_path should not be None when character_type is {}".format(character_type)
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)

        else:
            raise NotImplementedError
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                                                        idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


# def get_image_file_list(img_file):
#     imgs_lists = []
#     if img_file is None or not os.path.exists(img_file):
#         raise Exception("not found any img file in {}".format(img_file))
#
#     img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
#     if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
#         imgs_lists.append(img_file)
#     elif os.path.isdir(img_file):
#         for single_file in os.listdir(img_file):
#             file_path = os.path.join(img_file, single_file)
#             if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
#                 imgs_lists.append(file_path)
#     if len(imgs_lists) == 0:
#         raise Exception("not found any img file in {}".format(img_file))
#     imgs_lists = sorted(imgs_lists)
#     return imgs_lists


def rec_create_predictor(rec_model_dir, use_gpu, gpu_mem=500, use_tensorrt=False, use_fp16=False, max_batch_size=10,
                         enable_mkldnn=False, rec_batch_num=6):
    # if mode == "det":
    #     model_dir = args.det_model_dir
    # elif mode == 'cls':
    #     model_dir = args.cls_model_dir
    # elif mode == 'rec':
    #     model_dir = args.rec_model_dir
    # else:
    #     model_dir = args.e2e_model_dir

    if rec_model_dir is None:
        # logger.info("not find {} model file path {}".format(mode, model_dir))
        print("not find model file path {}".format(rec_model_dir))
        sys.exit(0)
    model_file_path = rec_model_dir + "/inference.pdmodel"
    params_file_path = rec_model_dir + "/inference.pdiparams"
    if not os.path.exists(model_file_path):
        # logger.info("not find model file path {}".format(model_file_path))
        print("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        # logger.info("not find params file path {}".format(params_file_path))
        "not find params file path {}".format(params_file_path)
        sys.exit(0)

    config = paddle.inference.Config(model_file_path, params_file_path)

    if use_gpu:
        config.enable_use_gpu(gpu_mem, 0)
        if use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=paddle.inference.PrecisionType.Half
                if use_fp16 else paddle.inference.PrecisionType.Float32,
                max_batch_size=max_batch_size)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(6)
        if enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            #  TODO LDOUBLEV: fix mkldnn bug when bach_size  > 1
            # config.set_mkldnn_op({'conv2d', 'depthwise_conv2d', 'pool2d', 'batch_norm'})
            rec_batch_num = 1

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.switch_use_feed_fetch_ops(False)

    # create predictor
    predictor = paddle.inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors, rec_batch_num


class TextRecognizer(object):
    def __init__(self, model_dir="./inference/rec_crnn", use_gpu=True, enable_mkldnn=False,
                 rec_image_shape=None, rec_char_type='ch', rec_char_dict_path='./ppocr/utils/ppocr_keys_v1.txt',
                 rec_batch_num=6, max_text_length=40, rec_algorithm='CRNN', use_space_char=False):
        # self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        if rec_image_shape is None:
            self.rec_image_shape = [3, 32, 320]
        # self.rec_image_shape = [3, 32, 100]
        # self.character_type = args.rec_char_type
        self.character_type = rec_char_type
        # self.rec_batch_num = args.rec_batch_num
        self.rec_batch_num = rec_batch_num
        # self.rec_algorithm = args.rec_algorithm
        self.rec_algorithm = rec_algorithm
        # self.max_text_length = args.max_text_length
        self.max_text_length = max_text_length
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        postprocess_params = {
            'name': 'CTCLabelDecode',
            # "character_type": args.rec_char_type,
            # "character_dict_path": args.rec_char_dict_path,
            # "use_space_char": args.use_space_char
            "character_type": rec_char_type,
            "character_dict_path": rec_char_dict_path,
            "use_space_char": use_space_char
        }
        # if self.rec_algorithm == "SRN":
        #     postprocess_params = {
        #         'name': 'SRNLabelDecode',
        #         "character_type": args.rec_char_type,
        #         "character_dict_path": args.rec_char_dict_path,
        #         "use_space_char": args.use_space_char
        #     }
        # elif self.rec_algorithm == "RARE":
        #     postprocess_params = {
        #         'name': 'AttnLabelDecode',
        #         "character_type": args.rec_char_type,
        #         "character_dict_path": args.rec_char_dict_path,
        #         "use_space_char": args.use_space_char
        #     }
        # self.postprocess_op = build_post_process(postprocess_params)
        self.postprocess_op = eval(postprocess_params['name'])(**postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.rec_batch_num = \
            rec_create_predictor(rec_model_dir=self.model_dir, use_gpu=self.use_gpu,
                                 enable_mkldnn=self.enable_mkldnn, rec_batch_num=self.rec_batch_num)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    # def resize_norm_img_srn(self, img, image_shape):
    #     imgC, imgH, imgW = image_shape
    #
    #     img_black = np.zeros((imgH, imgW))
    #     im_hei = img.shape[0]
    #     im_wid = img.shape[1]
    #
    #     if im_wid <= im_hei * 1:
    #         img_new = cv2.resize(img, (imgH * 1, imgH))
    #     elif im_wid <= im_hei * 2:
    #         img_new = cv2.resize(img, (imgH * 2, imgH))
    #     elif im_wid <= im_hei * 3:
    #         img_new = cv2.resize(img, (imgH * 3, imgH))
    #     else:
    #         img_new = cv2.resize(img, (imgW, imgH))
    #
    #     img_np = np.asarray(img_new)
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    #     img_black[:, 0:img_np.shape[1]] = img_np
    #     img_black = img_black[:, :, np.newaxis]
    #
    #     row, col, c = img_black.shape
    #     c = 1
    #
    #     return np.reshape(img_black, (c, row, col)).astype(np.float32)

    # def srn_other_inputs(self, image_shape, num_heads, max_text_length):
    #
    #     imgC, imgH, imgW = image_shape
    #     feature_dim = int((imgH / 8) * (imgW / 8))
    #
    #     encoder_word_pos = np.array(range(0, feature_dim)).reshape(
    #         (feature_dim, 1)).astype('int64')
    #     gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
    #         (max_text_length, 1)).astype('int64')
    #
    #     gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
    #     gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
    #         [-1, 1, max_text_length, max_text_length])
    #     gsrm_slf_attn_bias1 = np.tile(
    #         gsrm_slf_attn_bias1,
    #         [1, num_heads, 1, 1]).astype('float32') * [-1e9]
    #
    #     gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
    #         [-1, 1, max_text_length, max_text_length])
    #     gsrm_slf_attn_bias2 = np.tile(
    #         gsrm_slf_attn_bias2,
    #         [1, num_heads, 1, 1]).astype('float32') * [-1e9]
    #
    #     encoder_word_pos = encoder_word_pos[np.newaxis, :]
    #     gsrm_word_pos = gsrm_word_pos[np.newaxis, :]
    #
    #     return [
    #         encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
    #         gsrm_slf_attn_bias2
    #     ]

    # def process_image_srn(self, img, image_shape, num_heads, max_text_length):
    #     norm_img = self.resize_norm_img_srn(img, image_shape)
    #     norm_img = norm_img[np.newaxis, :]
    #
    #     [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
    #         self.srn_other_inputs(image_shape, num_heads, max_text_length)
    #
    #     gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
    #     gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
    #     encoder_word_pos = encoder_word_pos.astype(np.int64)
    #     gsrm_word_pos = gsrm_word_pos.astype(np.int64)
    #
    #     return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
    #             gsrm_slf_attn_bias2)

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                # if self.rec_algorithm != "SRN":
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
                # else:
                # norm_img = self.process_image_srn(img_list[indices[ino]],
                #                                   self.rec_image_shape, 8,
                #                                   self.max_text_length)
                # encoder_word_pos_list = []
                # gsrm_word_pos_list = []
                # gsrm_slf_attn_bias1_list = []
                # gsrm_slf_attn_bias2_list = []
                # encoder_word_pos_list.append(norm_img[1])
                # gsrm_word_pos_list.append(norm_img[2])
                # gsrm_slf_attn_bias1_list.append(norm_img[3])
                # gsrm_slf_attn_bias2_list.append(norm_img[4])
                # norm_img_batch.append(norm_img[0])
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            # if self.rec_algorithm == "SRN":
            #     starttime = time.time()
            #     encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
            #     gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
            #     gsrm_slf_attn_bias1_list = np.concatenate(
            #         gsrm_slf_attn_bias1_list)
            #     gsrm_slf_attn_bias2_list = np.concatenate(
            #         gsrm_slf_attn_bias2_list)
            #
            #     inputs = [
            #         norm_img_batch,
            #         encoder_word_pos_list,
            #         gsrm_word_pos_list,
            #         gsrm_slf_attn_bias1_list,
            #         gsrm_slf_attn_bias2_list,
            #     ]
            #     input_names = self.predictor.get_input_names()
            #     for i in range(len(input_names)):
            #         input_tensor = self.predictor.get_input_handle(input_names[
            #             i])
            #         input_tensor.copy_from_cpu(inputs[i])
            #     self.predictor.run()
            #     outputs = []
            #     for output_tensor in self.output_tensors:
            #         output = output_tensor.copy_to_cpu()
            #         outputs.append(output)
            #     preds = {"predict": outputs[2]}
            # else:
            starttime = time.time()
            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.run()

            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            preds = outputs[0]
            self.predictor.try_shrink_memory()
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            elapse += time.time() - starttime
        return rec_res, elapse


class TextSystem(object):
    def __init__(self, det_model_dir="./inference/ch_ppocr_server/ch_ppocr_server_v2.0_det_infer",
                 rec_model_dir="./inference/ch_ppocr_server/ch_ppocr_server_v2.0_rec_infer", char_type='ch',
                 char_dict_path='./ppocr/utils/ppocr_keys_v1.txt', use_gpu=True, enable_mkldnn=False,
                 drop_score=0.5, db_merge_boxes=False, db_box_thresh=0.5, db_unclip_ratio=1.6):

        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        self.text_detector = TextDetector(model_dir=det_model_dir, use_gpu=self.use_gpu,
                                          enable_mkldnn=self.enable_mkldnn,
                                          det_db_box_thresh=db_box_thresh, det_db_unclip_ratio=db_unclip_ratio)
        # self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = TextRecognizer(model_dir=rec_model_dir, use_gpu=self.use_gpu,
                                              enable_mkldnn=self.enable_mkldnn,
                                              rec_char_type=char_type, rec_char_dict_path=char_dict_path)
        # self.text_recognizer = predict_rec.TextRecognizer(args)
        # self.use_angle_cls = args.use_angle_cls
        self.drop_score = drop_score
        self.db_merge_boxes = db_merge_boxes
        # if self.use_angle_cls:
        #     self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            # logger.info(bno, rec_res[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        # logger.info("dt_boxes num : {}, elapse : {}".format(
        #     len(dt_boxes), elapse))
        print("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        if self.db_merge_boxes:
            dt_boxes = my_sorted_boxes(dt_boxes)
        else:
            dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        # if self.use_angle_cls:
        #     img_crop_list, angle_list, elapse = self.text_classifier(
        #         img_crop_list)
        #     logger.info("cls num  : {}, elapse : {}".format(
        #         len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        # logger.info("rec_res num  : {}, elapse : {}".format(
        #     len(rec_res), elapse))
        print("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def my_sorted_boxes(dt_boxes):
    """
    merge boxes to rows for rec, only used for almost upright images with regular dt_boxes
    #  TODO: use ML methods to optimize branching processes
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        merged boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    tops = [_boxes[0][0][1], _boxes[0][1][1]]
    bottoms = [_boxes[0][2][1], _boxes[0][3][1]]
    lefts = [_boxes[0][0][0], _boxes[0][3][0]]
    rights = [_boxes[0][1][0], _boxes[0][2][0]]
    t, b, l, r = 0, 0, 0, 0
    rows = []
    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10:  # 分行阈值待优化。。。
            tops.extend([_boxes[i + 1][0][1], _boxes[i + 1][1][1]])
            bottoms.extend([_boxes[i + 1][2][1], _boxes[i + 1][3][1]])
            lefts.extend([_boxes[i + 1][0][0], _boxes[i + 1][3][0]])
            rights.extend([_boxes[i + 1][1][0], _boxes[i + 1][2][0]])
        else:
            # rows.append([[np.min(np.asarray(lefts)), np.min(np.asarray(tops))],
            #              [np.max(np.asarray(rights)), np.min(np.asarray(tops))],
            #              [np.max(np.asarray(rights)), np.max(np.asarray(bottoms))],
            #              [np.min(np.asarray(lefts)), np.max(np.asarray(bottoms))]])
            t = np.min(np.asarray(tops))
            b = np.max(np.asarray(bottoms))
            l = np.min(np.asarray(lefts))
            r = np.max(np.asarray(rights))
            rows.append([[l, t], [r, t], [r, b], [l, b]])
            tops = [_boxes[i + 1][0][1], _boxes[i + 1][1][1]]
            bottoms = [_boxes[i + 1][2][1], _boxes[i + 1][3][1]]
            lefts = [_boxes[i + 1][0][0], _boxes[i + 1][3][0]]
            rights = [_boxes[i + 1][1][0], _boxes[i + 1][2][0]]
    # rows.append([[np.min(np.asarray(lefts)), np.min(np.asarray(tops))],
    #              [np.max(np.asarray(rights)), np.min(np.asarray(tops))],
    #              [np.max(np.asarray(rights)), np.max(np.asarray(bottoms))],
    #              [np.min(np.asarray(lefts)), np.max(np.asarray(bottoms))]])
    t = np.min(np.asarray(tops))
    b = np.max(np.asarray(bottoms))
    l = np.min(np.asarray(lefts))
    r = np.max(np.asarray(rights))
    rows.append([[l, t], [r, t], [r, b], [l, b]])

    return np.asarray(rows)


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][
            1]) ** 2)
        box_width = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][
            1]) ** 2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def main(image_dir="./predict_data/ch_ppocr_server", process_id=0, total_process_num=1,
         det_model="./inference/ch_ppocr_server/ch_ppocr_server_v2.0_det_infer",
         rec_model="./inference/ch_ppocr_server/ch_ppocr_server_v2.0_rec_infer",
         char_type='ch', char_dict='./ppocr/utils/ppocr_keys_v1.txt',
         use_gpu=True, enable_mkldnn=False, drop_score=0.5, merge_boxes=False,
         db_box_thresh=0.5, db_unclip_ratio=1.6,
         is_visualize=True, vis_font_path="./doc/fonts/simfang.ttf"):
    """
        主函数（接口）
        参数:
            image_dir：输入图片文件（或文件夹），建议采用jpg格式，图片尺寸尽量满足长边的值不大于960
            process_id：进程号
            total_process_num：线程数
            det_model：文本检测推理模型
            rec_model：文本识别推理模型
            char_type：语种
            char_dict：字典
            use_gpu：使用gpu推理
            enable_mkldnn：使用mkldnn加速（cpu推理时）
            drop_score：置信度
            merge_boxes：合并检测框
            db_box_thresh：检测框阈值
            db_unclip_ratio：检测框扩张系数
            is_visualize：可视化
            vis_font_path：可视化所需字体文件
    """
    image_file_list = get_image_file_list(image_dir)
    image_file_list = image_file_list[process_id::total_process_num]
    # 构建文本串联推理模型
    text_sys = TextSystem(det_model_dir=det_model, rec_model_dir=rec_model, char_type=char_type,
                          char_dict_path=char_dict, use_gpu=use_gpu, enable_mkldnn=enable_mkldnn,
                          drop_score=drop_score, db_merge_boxes=merge_boxes,
                          db_box_thresh=db_box_thresh, db_unclip_ratio=db_unclip_ratio)
    font_path = vis_font_path
    # drop_score = drop_score
    for image_file in image_file_list:
        # img, flag = check_and_read_gif(image_file)
        # if not flag:
        for i in range(100):
            img = cv2.imread(image_file)
            if img is None:
                # logger.info("error in loading image:{}".format(image_file))
                print("error in loading image:{}".format(image_file))
                continue
            starttime = time.time()
            dt_boxes, rec_res = text_sys(img)
            elapse = time.time() - starttime
            # logger.info("Predict time of %s: %.3fs" % (image_file, elapse))
            print("Predict time of %s: %.3fs" % (image_file, elapse))
        # img = cv2.imread(image_file)
        # if img is None:
        #     # logger.info("error in loading image:{}".format(image_file))
        #     print("error in loading image:{}".format(image_file))
        #     continue
        # starttime = time.time()
        # dt_boxes, rec_res = text_sys(img)
        # elapse = time.time() - starttime
        # # logger.info("Predict time of %s: %.3fs" % (image_file, elapse))
        # print("Predict time of %s: %.3fs" % (image_file, elapse))

        for text, score in rec_res:
            # logger.info("{}, {:.3f}".format(text, score))
            print("{}, {:.3f}".format(text, score))
        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]
            # 可视化过程
            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            # logger.info("The visualized image saved in {}".format(
            #     os.path.join(draw_img_save, os.path.basename(image_file))))
            print("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))


if __name__ == "__main__":

    # 有关线程的部分先保持默认不动，这块看不懂，待优化。。。
    use_mp = False  # 是否开启多线程
    process_num = 1  # 线程数
    if use_mp:
        p_list = []
        total_process_num = process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    # 关注下面的main函数调用即可。。。

    else:
        main(image_dir="./predict_data/ch_ppocr_server/our_3.jpg",
             det_model="./inference/ch_ppocr_server/ch_ppocr_mobile_v2.0_det_prune_infer",
             rec_model="./inference/ch_ppocr_server/ch_ppocr_server_v2.0_rec_infer",
             use_gpu=False, enable_mkldnn=True, merge_boxes=True)
