from my_dataset_coco import CocoKeypoint
import transforms
import json
import torch
import numpy as np
if __name__ == '__main__':
    # 读取kp的关键点基本信息，名称、权重等
    with open("../person_keypoints.json", "r") as f:
        person_kps_info = json.load(f)

    fixed_size = [256, 192]
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((17,))
    data_transform = {
        "train": transforms.Compose([
            transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),# 随机截取上下一半
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),# 仿射变换
            transforms.RandomHorizontalFlip(0.5, person_kps_info["flip_pairs"]), # 随机水平翻转
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            # transforms.ToTensor(), # 把Image通道h*w*c转化为c*h*w，并把像素/255，对应的heatmap应该没有转变，生成的时候就是通道在前面（keynum）
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # img归一化，mean和std是官方统计的结果
        ])
    }
    train = CocoKeypoint("../data/coco2017/", dataset="train",transforms=data_transform['train'])
    img,tag = train.__getitem__(0)
    print(img)


    # a = torch.tensor(range(8)).resize(2,2,2).float()
    # print(a)
    # avg = a.mean([1,2])
    # print(avg)