import cv2
from glob import glob
import numpy as np
from pathlib import Path

def resize_with_padding(image, target_size=(512, 512)):
    # 원본 이미지의 크기
    h, w, _ = image.shape
    target_w, target_h = target_size
    base_pic = np.zeros((target_h, target_w, 3), np.uint8)
    ratio_h = target_h / h
    ratio_w = target_w / w
    if ratio_w < ratio_h:
        new_size = (int(w * ratio_w), int(h * ratio_w))

    else:
        new_size = (int(w * ratio_h), int(h * ratio_h))
    new_w, new_h = new_size
    image = cv2.resize(image, dsize=new_size)
    base_pic[int(target_h / 2 - new_h / 2) : int(target_h / 2 + new_h / 2), int(target_w / 2 - new_w / 2) : int(target_w / 2 + new_w / 2), :,] = image
    return base_pic

def main():
    files = glob("/home/kangnam/datasets/images/1_kangnam_diffusion/*.png")
    save_dir = Path("/home/kangnam/datasets/images/kangnam_resize/")
    
    for file in files:
        image = cv2.imread(file)
        image = resize_with_padding(image, target_size=(256, 256))
        
        name = Path(file).name
        save_path = save_dir.joinpath(name)
        cv2.imwrite(str(save_path.resolve()), image)
        
        
if __name__ == "__main__":
    main()