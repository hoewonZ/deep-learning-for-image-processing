import os
import copy

import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO


class CocoKeypoint(data.Dataset):
    def __init__(self,
                 root,
                 dataset="train",
                 years="2017",
                 transforms=None,
                 det_json_path=None,
                 fixed_size=(256, 192)):
        super().__init__()
        '''
            1：定位训练集和标记文件位置。
        '''
        #train 、val 是跟data/coco2017/annotation/ 下person_keypoints_xxx.json来的
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        # keypoint train 文件
        anno_file = f"person_keypoints_{dataset}{years}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        # img_root = ./data/coco2017/train2017 照片文件夹
        self.img_root = os.path.join(root, f"{dataset}{years}")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        # annotation 文件路径 ./data/coco2017/annotations/person_keypoints_train2017.json
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.fixed_size = fixed_size
        self.mode = dataset
        self.transforms = transforms

        # 加载person_keypoints_train2017.json中的数据
        self.coco = COCO(self.anno_path)

        # 读取Image 在json文件中id编号，不是image的path
        img_ids = list(sorted(self.coco.imgs.keys()))

        # det应该是检测任务中，或者为了区分读img和读ann的对象不同，代码上区分开。在kp任务中应该是同一个对象
        if det_json_path is not None:
            det = self.coco.loadRes(det_json_path)
        else:
            det = self.coco
        # 把json中检测相关的img 、ann信息放到列表里，读取这个列表中的数据进行训练
        self.valid_person_list = []
        obj_idx = 0
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            # 一张图片可能有注释文件中标记了关键点信息。
            ann_ids = det.getAnnIds(imgIds=img_id)
            anns = det.loadAnns(ann_ids)
            for ann in anns:
                # only save person class
                if ann["category_id"] != 1:
                    print(f'warning: find not support id: {ann["category_id"]}, only support id: 1 (person)')
                    continue

                # COCO_val2017_detections_AP_H_56_person.json文件中只有det信息，没有keypoint信息，跳过检查
                if det_json_path is None:
                    # skip objs without keypoints annotation
                    # 没有keypoints属性和为0的ann直接pass
                    if "keypoints" not in ann:
                        continue
                    if max(ann["keypoints"]) == 0:
                        continue

                xmin, ymin, w, h = ann['bbox']
                # Use only valid bounding boxes
                if w > 0 and h > 0:
                    info = {
                        "box": [xmin, ymin, w, h],
                        "image_path": os.path.join(self.img_root, img_info["file_name"]),
                        "image_id": img_id,
                        "image_width": img_info['width'],
                        "image_height": img_info['height'],
                        "obj_origin_hw": [h, w],
                        "obj_index": obj_idx,
                        "score": ann["score"] if "score" in ann else 1.
                    }

                    # COCO_val2017_detections_AP_H_56_person.json文件中只有det信息，没有keypoint信息，跳过
                    if det_json_path is None:
                        # 把json中17*3个1维的关键点转换为（17，3）的关键点，
                        keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                        # 取第3列
                        visible = keypoints[:, 2]
                        # 取前2列
                        keypoints = keypoints[:, :2]
                        # 把keypoints坐标和visible放入info对象里
                        info["keypoints"] = keypoints
                        info["visible"] = visible
                    # info放入list中
                    self.valid_person_list.append(info)
                    obj_idx += 1

            # 为了测试，数据集只读取第一个就跳出
            # break

    def __getitem__(self, idx):
        # 不改变原数据
        target = copy.deepcopy(self.valid_person_list[idx])

        # opencv 读取图片时通道是BGR顺序，我们习惯用RGB顺序进行训练和预测，因此用cvtColor转换一下通道顺序而已。PIL读取就是RGB顺序。
        image = cv2.imread(target["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('ori',image)
        if self.transforms is not None:
            image, person_info = self.transforms(image, target)

        # transformer和归一化等之后的图像和target

        return image, target

    def __len__(self):
        return len(self.valid_person_list)

    '''
    collate_fn(batch) 只一个dataloader会调用的方法，用来处理batchsize对应生成的dataset列表。dataloader生成的数据不是tensor的，
    因此自定义一个回调（当然dataloader本身也有一个collate_fn）把数据展开成tensor格式的多个真实数据。
    '''
    @staticmethod
    def collate_fn(batch):
        # *解构，相当于把batch这个tuple拆成一个一个传给方法。dataset每次getitem返回image和target，所以 batch应该是一个（（image,target），(...,...),...）的东西。
        # 把原tuple分成两个tuple
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple


if __name__ == '__main__':
    train = CocoKeypoint("/data/coco2017/", dataset="val")
    print(len(train))
    t = train[0]
    print(t)
