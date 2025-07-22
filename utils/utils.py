import os
# import ensemble_boxes
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
import torch.serialization  



def normalize_boxes(boxes, img_size, invert=False):
    if invert:
        boxes[..., 0] = boxes[..., 0] * img_size[0]
        boxes[..., 1] = boxes[..., 1] * img_size[1]
        boxes[..., 2] = boxes[..., 2] * img_size[0]
        boxes[..., 3] = boxes[..., 3] * img_size[1]
    else:
        boxes[..., 0] = boxes[..., 0] / img_size[0]
        boxes[..., 1] = boxes[..., 1] / img_size[1]
        boxes[..., 2] = boxes[..., 2] / img_size[0]
        boxes[..., 3] = boxes[..., 3] / img_size[1]
    return boxes




def bounding_boxes(v_boxes, v_labels, v_scores, log_width, log_height, class_id_to_label, score_threshold):
    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        # get coordinates and labels

        if v_scores[b_i] < score_threshold or int(v_labels[b_i]) == -1:  # stop when below this threshold, scores in descending order
            break

        caption = "%s" % (class_id_to_label[int(v_labels[b_i])])
        if v_scores[b_i] <= 1:
            caption = "%s (%.3f)" % (class_id_to_label[int(v_labels[b_i])], v_scores[b_i])
        # from xyxy
        box_data = {"position" : {
            "minX" : int(box[0]),
            "minY" : int(box[1]),
            "maxX" : int(box[2]),
            "maxY" : int(box[3])},
            "class_id" : int(v_labels[b_i]),
            # optionally caption each box with its class and score
            "box_caption" : caption,
            "domain" : "pixel",
            "scores" : { "score" : int(v_scores[b_i]*100) }}

        all_boxes.append(box_data)
    return all_boxes


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

'''def draw_pred_boxes(img_path, predicted_boxes, class_id_to_color, output_path):
    import cv2
    image = cv2.imread(str(img_path))
    for box in predicted_boxes:
        xmin = box['position']['minX']
        ymin = box['position']['minY']
        xmax = box['position']['maxX']
        ymax = box['position']['maxY']
        label = box['box_caption']
        class_id = box['class_id']
        color = class_id_to_color.get(class_id, (128, 128, 128))

        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(image, (xmin, ymin - ret[1] - baseline), (xmin + ret[0], ymin), color, -1)
        cv2.putText(image, label, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(output_path, image)'''
    
def draw_pred_boxes(img_path, predicted_boxes, class_id_to_color, class_id_to_label, output_path):
    import cv2
    image = cv2.imread(str(img_path))
    for box in predicted_boxes:
        xmin = box['position']['minX']
        ymin = box['position']['minY']
        xmax = box['position']['maxX']
        ymax = box['position']['maxY']
        class_id = box['class_id']
        color = class_id_to_color.get(class_id, (128, 128, 128))

        # 두 줄 caption 처리
        caption = box.get('box_caption', [class_id_to_label.get(class_id, 'unknown')])
        if not isinstance(caption, list):
            caption = [caption]  # fallback 처리

        font_scale = 0.5
        thickness = 1

        # 첫 줄 텍스트 크기 계산 (가장 긴 줄 기준)
        max_width = 0
        total_height = 0
        text_sizes = []
        for line in caption:
            (w, h), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_sizes.append(((w, h), base))
            max_width = max(max_width, w)
            total_height += h + base

        # 텍스트 배경 박스
        text_y = max(ymin - total_height, 0)
        cv2.rectangle(image, (xmin, text_y), (xmin + max_width, text_y + total_height), color, -1)

        # 각 줄 텍스트 출력
        line_y = text_y
        for i, line in enumerate(caption):
            (w, h), base = text_sizes[i]
            cv2.putText(image, line, (xmin, line_y + h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            line_y += h + base

        # 바운딩 박스
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)

    cv2.imwrite(output_path, image)

    
def visualize_detections(dataset, detections, target, output_dir, args, split='val', score_threshold=0.5, img_tensor=None):
    #os.makedirs(output_dir, exist_ok=True)
    
    # 클래스 라벨과 고정 색상 정의
    class_id_to_label = {int(i): str(i) for i in range(1, args.num_classes + 1)}
    class_id_to_label.update({1: "person", 2: "bicycle", 3: "car"})
    class_id_to_color = {
        1: (255, 0, 0),   # person - blue
        2: (0, 255, 0),   # bicycle - green
        3: (0, 0, 255),   # car - red
    }
    '''class_id_to_label.update({1: "people", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck", 6: "lamp"})
    class_id_to_color = {
    1: (255, 0, 0),     
    2: (0, 255, 0),     
    3: (0, 0, 255),     
    4: (255, 255, 0),   
    5: (255, 165, 0),   
    6: (128, 0, 128),   
    }'''
    '''class_id_to_label.update({1: "board", 2: "fire", 3: "light", 4: "utensil", 5: "person"})

    class_id_to_color = {
    1: (255, 165, 0),   # board - 주황
    2: (255, 0, 0),     # fire - 빨강
    3: (0, 255, 255),   # light - 밝은 하늘색
    4: (0, 255, 0),     # utensil - 초록
    5: (0, 0, 255),     # person - 파랑
    }'''
    detections = detections.detach().cpu().numpy()
    img_indices = target['img_idx'].cpu().numpy()
    img_scales = target['img_scale'].cpu().numpy()

    for i, (img_idx, img_dets, img_scale) in enumerate(zip(img_indices, detections, img_scales)):
        img_info = dataset.parser.img_infos[img_idx]
        img_name = img_info['file_name']

        # 예측 박스 필터링 및 변환
        predicted_boxes = []
        for j in range(img_dets.shape[0]):
            if img_dets[j, 4] < score_threshold:
                continue
            class_id = int(img_dets[j, 5])
            score_val = img_dets[j, 4]
            xmin, ymin, xmax, ymax = map(int, img_dets[j, 0:4])
            #caption = f"{class_id_to_label.get(class_id, 'unknown')} ({score_val:.3f})"
            caption = [
                class_id_to_label.get(class_id, 'unknown'),
                f"({score_val:.3f})"
            ]
            predicted_boxes.append({
                "position": {"minX": xmin, "minY": ymin, "maxX": xmax, "maxY": ymax},
                "class_id": class_id,
                "box_caption": caption,
            })

        # 이미지 경로 및 출력 경로 정의
        filename_thermal = dataset.thermal_data_dir / img_name
        filename_rgb = dataset.rgb_data_dir / img_name.replace('_PreviewData.jpg', '_RGB.jpg')

        # 저장 경로는 동일 output_dir 아래
        basename = img_name.replace('.jpg', '')# 확장자 제거
        thermal_out = os.path.join(output_dir, f"{basename}_thermal_gate.png")
        rgb_out     = os.path.join(output_dir, f"{basename}_rgb_gate.png")
        '''filename_thermal = dataset.thermal_data_dir / img_name
        filename_rgb = dataset.rgb_data_dir / img_name

        # 확장자를 유지한 채 _thermal_pred, _rgb_pred 추가
        basename = img_name.replace('.png', '')  # '00001' 형태
        thermal_out = os.path.join(output_dir, f"{basename}_thermal_pred.png")
        rgb_out = os.path.join(output_dir, f"{basename}_rgb_pred.png")'''

        # 박스 시각화 및 저장
        #draw_pred_boxes(filename_thermal, predicted_boxes, class_id_to_color, thermal_out)
        #draw_pred_boxes(filename_rgb, predicted_boxes, class_id_to_color, rgb_out)
        draw_pred_boxes(filename_thermal, predicted_boxes, class_id_to_color, class_id_to_label, thermal_out)
        draw_pred_boxes(filename_rgb, predicted_boxes, class_id_to_color, class_id_to_label, rgb_out)
       
def visualize_target(dataset, target, output_dir, args, split='val', img_tensor=None):
    #os.makedirs(output_dir, exist_ok=True)

    # 클래스 라벨과 색상 정의
    class_id_to_label = {int(i): str(i) for i in range(1, args.num_classes + 1)}
    '''class_id_to_label.update({1: "people", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck", 6: "lamp"})
    class_id_to_color = {
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        5: (255, 165, 0),
        6: (128, 0, 128),
    }'''
    class_id_to_label.update({1: "person", 2: "bicycle", 3: "car"})
    class_id_to_color = {
        1: (255, 0, 0),   # person - blue
        2: (0, 255, 0),   # bicycle - green
        3: (0, 0, 255),   # car - red
    }
    '''class_id_to_label.update({1: "board", 2: "fire", 3: "light", 4: "utensil", 5: "person"})
    class_id_to_color = {
    1: (0, 128, 255),   
    2: (255, 165, 0),     
    3: (255, 255, 0),   
    4: (0, 255, 0),     
    5: (255, 0, 0),   
    }
'''
    img_indices = target['img_idx'].cpu().numpy()
    bboxes = target['bbox'].cpu().numpy()
    clses = target['cls'].cpu().numpy()
    scores = target['scores'].cpu().numpy() if 'scores' in target else np.ones_like(clses)  # GT 점수 1로 설정
    img_scales = target['img_scale'].cpu().numpy()

    for i, (img_idx, bbox, cls, img_scale, score) in enumerate(zip(img_indices, bboxes, clses, img_scales, scores)):
        img_info = dataset.parser.img_infos[img_idx]
        img_name = img_info['file_name']

        # 바운딩 박스 스케일 조정 및 좌표 변환
        if img_tensor is None:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]] * img_scale
            filename = dataset.thermal_data_dir / img_name
            raw_image_path = str(filename)
        else:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]]
            raw_image_path = None

        # box 리스트 구성
        gt_boxes = []
        for j in range(bbox.shape[0]):
            class_id = int(cls[j])
            if class_id == -1:
                continue  # 무효 클래스 생략

            xmin, ymin, xmax, ymax = map(int, bbox[j][:4])
            caption = [class_id_to_label.get(class_id, 'unknown')]  # ✅ 점수 제거
            gt_boxes.append({
                "position": {"minX": xmin, "minY": ymin, "maxX": xmax, "maxY": ymax},
                "class_id": class_id,
                "box_caption": caption,
            })

        # 시각화 및 저장
        if raw_image_path:
            filename_thermal = dataset.thermal_data_dir / img_name
            filename_rgb = dataset.rgb_data_dir / img_name.replace('_PreviewData.jpg', '_RGB.jpg')

            # 저장 경로는 동일 output_dir 아래
            basename = img_name.replace('.jpg', '').replace('.png', '')  # 확장자 제거
            thermal_out = os.path.join(output_dir, f"{basename}_thermal_gt.jpg")
            rgb_out     = os.path.join(output_dir, f"{basename}_rgb_gt.jpg")

            # 저장 실행 (변수명 일치)
            draw_pred_boxes(filename_thermal, gt_boxes, class_id_to_color, class_id_to_label, thermal_out)
            draw_pred_boxes(filename_rgb, gt_boxes, class_id_to_color, class_id_to_label, rgb_out)



def load_checkpoint_selective(net, snapshot, scene=None):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))

    '''with torch.serialization.safe_globals(["argparse.Namespace"]):
        checkpoint = torch.load(snapshot, map_location=torch.device('cpu'), weights_only=False)'''

    if 'state_dict' in checkpoint:
        net = state_restore_selective(net, checkpoint['state_dict'], scene)
    else:
        net = state_restore_selective(net, checkpoint, scene)

    return net


def state_restore_selective(net, loaded_dict, scene=None):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in loaded_dict:
        if 'cbam' in k:
            new_loaded_dict[k.replace('fusion', 'fusion'+str(scene))] = loaded_dict[k]
            print('successfully loaded ', k.replace('fusion', 'fusion'+str(scene)))
        if 'classifier' in k:
            new_loaded_dict[k] = loaded_dict[k]
            print('successfully loaded ', k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

'''def visualize_detections(dataset, detections, target, wandb, args, split='val', score_threshold=0.5):
    class_id_to_label = { int(i) : str(i) for i in range(1, args.num_classes + 1)}
    class_id_to_label.update({1: "person", 2: "bicycle", 3: "car"})
    #class_id_to_label.update({1: "people", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck", 6: "lamp"})
    detections = detections.detach().cpu().numpy()
    img_indices = target['img_idx'].cpu().numpy()
    bboxes = target['bbox'].cpu().numpy()
    clses = target['cls'].cpu().numpy()
    scores = target['scores'].cpu().numpy() if 'scores' in target else np.zeros_like(clses) + 1000
    img_scales = target['img_scale'].cpu().numpy()
    for img_idx, img_dets, bbox, cls, img_scale, score in zip(img_indices, detections, bboxes, clses, img_scales, scores):
        img_id = dataset.parser.img_ids[img_idx]
        img_info = dataset.parser.img_infos[img_idx]
        # yxyx to xyxy
        bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]] * img_scale
        predicted_boxes = bounding_boxes(
            v_boxes=img_dets[:, 0:4],
            v_labels=img_dets[:, 5],
            v_scores=img_dets[:, 4],
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=score_threshold)
        gt_boxes = bounding_boxes(
            v_boxes=bbox,
            v_labels=cls,
            v_scores=score,
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=0)
        filename = dataset.thermal_data_dir/img_info['file_name']
        filename_rgb = dataset.rgb_data_dir/img_info['file_name'].replace('_PreviewData.jpg', '_RGB.jpg')
        raw_image = Image.open(filename).convert('RGB')
        raw_image_rgb = Image.open(filename_rgb).convert('RGB')
        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = wandb.Image(raw_image, boxes = {"predictions": {"box_data": predicted_boxes, "class_labels" : class_id_to_label},
                                                    "gts": {"box_data": gt_boxes, "class_labels" : class_id_to_label}},
                                                    caption=str(filename))
        box_image_rgb = wandb.Image(raw_image_rgb, boxes = {"predictions": {"box_data": predicted_boxes, "class_labels" : class_id_to_label},
                                                    "gts": {"box_data": gt_boxes, "class_labels" : class_id_to_label}},
                                                   caption=str(filename_rgb))
        wandb.log({'thermal': box_image})
        wandb.log({'rgb': box_image_rgb})'''


class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.
        The total score is the sum of all the box scores.
    """
    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs[1]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs[1])
            index = ious.argmax()
            # if ious[0, index] > self.iou_threshold and model_outputs[0][index] == label:
            #     score = ious[0, index] + model_outputs['scores'][index]
            #     output = output + score
        return output