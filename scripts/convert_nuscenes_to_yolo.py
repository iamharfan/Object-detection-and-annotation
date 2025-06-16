import os
import cv2
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

# Define paths
NUSCENES_ROOT = '../datasets/nuscenes'
YOLO_ROOT = '../datasets/yolo-format'
CAMERA = 'CAM_FRONT'

# Classes to include (with YOLO class IDs)
CLASS_NAMES = {
    'car': 0,
    'pedestrian': 1,
    'truck': 2,
    'bus': 3,
    'motorcycle': 4,
    'bicycle': 5,
    'traffic_cone': 6,
    'barrier': 7,
    'construction_vehicle': 8,
    'trailer': 9
}

def get_2d_bbox(nusc, sample_token, cam_channel):
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data'][cam_channel])
    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    annotations = sample['anns']
    image_path = os.path.join(NUSCENES_ROOT, cam_data['filename'])

    img = cv2.imread(image_path)
    h, w, _ = img.shape
    boxes = []

    for ann_token in annotations:
        ann = nusc.get('sample_annotation', ann_token)
        category = ann['category_name'].split('.')[0]
        if category not in CLASS_NAMES:
            continue

        box = Box(ann['translation'], ann['size'], ann['rotation'])
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(np.linalg.inv(Quaternion(ego_pose['rotation']).rotation_matrix))
        box.translate(-np.array(cam_calib['translation']))
        box.rotate(np.linalg.inv(Quaternion(cam_calib['rotation']).rotation_matrix))

        corners = view_points(box.corners(), np.array(cam_calib['camera_intrinsic']), normalize=True)[:2]
        x_min, y_min = np.min(corners, axis=1)
        x_max, y_max = np.max(corners, axis=1)

        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        box_width = (x_max - x_min) / w
        box_height = (y_max - y_min) / h

        if 0 < x_center < 1 and 0 < y_center < 1 and box_width > 0 and box_height > 0:
            boxes.append([CLASS_NAMES[category], x_center, y_center, box_width, box_height])

    return boxes, image_path

def main():
    nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_ROOT, verbose=True)
    train_scenes = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0799', 'scene-0916', 'scene-0944', 'scene-0958', 'scene-0965', 'scene-1037']
    val_scenes = ['scene-0067', 'scene-0103']

    for sample in tqdm(nusc.sample):
        scene = nusc.get('scene', sample['scene_token'])['name']
        split = 'train' if scene in train_scenes else 'val' if scene in val_scenes else None
        if split is None:
            continue

        boxes, img_path = get_2d_bbox(nusc, sample['token'], CAMERA)
        if not boxes:
            continue

        # Copy image
        img_name = os.path.basename(img_path)
        out_img_path = os.path.join(YOLO_ROOT, f'images/{split}', img_name)
        out_lbl_path = os.path.join(YOLO_ROOT, f'labels/{split}', img_name.replace('.jpg', '.txt'))

        os.system(f'cp {img_path} {out_img_path}')

        # Write YOLO label
        with open(out_lbl_path, 'w') as f:
            for box in boxes:
                f.write(f"{box[0]} {' '.join(f'{x:.6f}' for x in box[1:])}\n")

    print("✅ Conversion complete.")

if __name__ == '__main__':
    from pyquaternion import Quaternion
    main()

