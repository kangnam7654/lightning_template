import numpy as np

# pseudo
def cutmix(image1, label1, image2, label2):
    width = image1.shape[0]
    height = image1.shape[1]
    random_lambda = np.random.uniform(0, 1)

    # equation -> w * h * lambda == r_w * r_h
    # adjust ratio
    random_width = np.random.uniform(0, width)
    random_height = (random_lambda * width * height) / random_width
    
    x1 = np.random.uniform(0, random_width)
    x2 = x1 + random_width
    y1 = np.random.uniform(0, random_height)
    y2 = y1 + random_height

    x1 = np.floor(x1)
    x2 = np.floor(x2)
    y1 = np.floor(y1)
    y2 = np.floor(y2)

    image1[x1:x2, y1:y2, :] = image2[x1:x2, y1:y2, :]
    lambda_adjusted = 1 - (x2 - x1) * (y2 - y1) / (width * height)
    label = lambda_adjusted * label1 + (1 - lambda_adjusted) * label2
    return image1, label