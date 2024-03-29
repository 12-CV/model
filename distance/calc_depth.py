import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from ultralytics import YOLO
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class DepthVisualizer:
    def __init__(self, img_path, outdir='./depth_vis', encoder='vits', pred_only=False, grayscale=False):
        self.img_path = img_path
        self.outdir = outdir
        self.encoder = encoder
        self.pred_only = pred_only
        self.grayscale = grayscale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("./model/yolow-l.pt") # yolo 모델의 경로를 입력
        self.depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(self.DEVICE).eval()
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        os.makedirs(self.outdir, exist_ok=True)

    def process_images(self):
        filenames = []
        if os.path.isfile(self.img_path):
            if self.img_path.endswith('txt'):
                with open(self.img_path, 'r') as f:
                    filenames = f.read().splitlines()
            else:
                filenames = [self.img_path]
        else:
            filenames = os.listdir(self.img_path)
            filenames = [os.path.join(self.img_path, filename) for filename in filenames if not filename.startswith('.')]
            filenames.sort()
        
        output_txt_file = "output.txt"
        with open(output_txt_file, "w") as f:
            for filename in tqdm(filenames):
                real_dist = filename.split("/")[-1].split(".")[0].split('_')[0]
                raw_image = cv2.imread(filename)
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
                h, w = image.shape[:2]
                image = self.transform({'image': image})['image']
                image = torch.from_numpy(image).unsqueeze(0).to(self.DEVICE)
                
                with torch.no_grad():
                    depth = self.depth_anything(image)
                
                depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
                depth_dist = depth.cpu().numpy().astype(np.uint8)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.cpu().numpy().astype(np.uint8)
                
                if self.grayscale:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                else:
                    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                
                if self.pred_only:
                    yolo_results = self.model(raw_image)
                    max_depth_per_box = {}
                    for r in yolo_results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            depth_in_box = depth_dist[y1:y2, x1:x2]
                            max_depth_per_box[str(box.xyxy[0])] = np.max(depth_in_box)
                    
                    for r in yolo_results:
                        boxes = r.boxes
                        max_depth_box = None
                        max_depth = -1
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            depth_in_box = depth_dist[y1:y2, x1:x2]
                            max_depth_in_box = np.max(depth_in_box)
                            if max_depth_in_box > max_depth:
                                max_depth = max_depth_in_box
                                max_depth_box = box
                        
                        if max_depth_box is not None:
                            x1, y1, x2, y2 = map(int, max_depth_box.xyxy[0])
                            distance = 0.01875 * max_depth**2 + -0.83062 * max_depth + 10.2852
                            cv2.putText(raw_image, f"{float(max_depth):.2f}"+ "  " +f"{float(real_dist)*0.01:.2f}" + 'm'+ "  " +f"{float(distance):.2f}" + 'm', (x1 - 30, y1 - 10), self.font, self.font_scale, (0, 255, 0), self.font_thickness)
                            cv2.rectangle(raw_image, (x1, y1), (x2, y2), (0,255,0), 2)
                    
                    cv2.imwrite(os.path.join(self.outdir, real_dist +'_viz.png'), raw_image)
                    f.write(f"{float(max_depth):.2f} {float(real_dist)*0.01:.2f} {float(distance):.2f}\n")
                    print("image saved")
                else:
                    # code for non-pred_only mode
                    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', default='./images', type=str)  # 이미지 데이터 경로 입력
    parser.add_argument('--outdir', type=str, default='./depth_vis')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    args = parser.parse_args()
    
    visualizer = DepthVisualizer(args.img_path, args.outdir, args.encoder, args.pred_only, args.grayscale)
    visualizer.process_images()
