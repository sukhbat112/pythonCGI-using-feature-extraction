import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

def resize_image(image, target_size=(256, 256)):
    resized_image = cv2.resize(image, target_size)
    return resized_image

def extract_color_histograms(image):
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    
    hist_rgb = cv2.calcHist([rgb], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    hist_luv = cv2.calcHist([luv], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    
    hist_rgb = cv2.normalize(hist_rgb, hist_rgb).flatten()
    hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
    hist_luv = cv2.normalize(hist_luv, hist_luv).flatten()
    
    return hist_rgb, hist_hsv, hist_luv

def extract_color_histograms2(image):
    
    height, width = image.shape[:2]
    
    # 2x2 regions
    regions = [
        (0, 0, width // 2, height // 2),  # top-left
        (width // 2, 0, width, height // 2),  # top-right
        (0, height // 2, width // 2, height),  # bottom-left
        (width // 2, height // 2, width, height)  # bottom-right
    ]
    
    # lists to store histograms
    hist_rgb_parts = []
    hist_hsv_parts = []
    hist_luv_parts = []
    
    # ヒストグラム計算
    for (x1, y1, x2, y2) in regions:
        region = image[y1:y2, x1:x2]
        hist_rgb, hist_hsv, hist_luv = extract_color_histograms(region)
        hist_rgb_parts.append(hist_rgb)
        hist_hsv_parts.append(hist_hsv)
        hist_luv_parts.append(hist_luv)
    
    # 各領域のヒストグラムを連結する
    hist_rgb = np.concatenate(hist_rgb_parts)
    hist_hsv = np.concatenate(hist_hsv_parts)
    hist_luv = np.concatenate(hist_luv_parts)
    
    return hist_rgb, hist_hsv, hist_luv


def extract_color_histograms3(image):
    
    height, width = image.shape[:2]
    
    # 3x3 regions
    regions = []
    region_width = width // 3
    region_height = height // 3
    for i in range(3):
        for j in range(3):
            regions.append((j * region_width, i * region_height, (j + 1) * region_width, (i + 1) * region_height))
    
    # lists to store histograms
    hist_rgb_parts = []
    hist_hsv_parts = []
    hist_luv_parts = []
    
    # 各領域のヒストグラム
    for (x1, y1, x2, y2) in regions:
        region = image[y1:y2, x1:x2]
        hist_rgb, hist_hsv, hist_luv = extract_color_histograms(region)
        hist_rgb_parts.append(hist_rgb)
        hist_hsv_parts.append(hist_hsv)
        hist_luv_parts.append(hist_luv)
    
    # 各領域のヒストグラムを連結する
    hist_rgb = np.concatenate(hist_rgb_parts)
    hist_hsv = np.concatenate(hist_hsv_parts)
    hist_luv = np.concatenate(hist_luv_parts)
    
    return hist_rgb, hist_hsv, hist_luv


def extract_gabor_features(image):
    #grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gabor filter parameters
    ksize = 31
    sigma = 5
    theta = np.pi / 4
    lambd = 10
    gamma = 0.5
    
    # Gabor filter
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    
    # apply Gabor filter to image
    gabor_img = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
    
    # calculate Gabor features
    hist = cv2.calcHist([gabor_img], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

def extract_dcnn_features(image):
    # define transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    # VGG16 model
    vgg16 = models.vgg16(pretrained=True)
    
    # remove the last fc layer     
    vgg16 = torch.nn.Sequential(
    vgg16.features,
    vgg16.avgpool,
    torch.nn.Flatten(),
    *list(vgg16.classifier.children())[:-1]  
    )

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #vgg16 = vgg16.to(device)
            
    # set model to evaluation mode
    vgg16.eval()
    
    # forward pass
    with torch.no_grad():
        output = vgg16(input_batch)
    
    # flatten the output tensor
    #output = output.view(output.size(0), -1)
    
    # L2 normalization
    output /= torch.sqrt(torch.sum(output**2))
    
    return output.numpy()


all_features_rgb = []
all_features_hsv = []
all_features_luv = []

all_features_rgb2 = []
all_features_hsv2 = []
all_features_luv2 = []
all_features_rgb3 = []
all_features_hsv3 = []
all_features_luv3 = []

all_features_gabor = []
all_features_dcnn = []

image_dir = 'img_kadai3a'
image_files = []

#create list containing all filenames in ascending order
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_files.append(os.path.join(image_dir, filename))

#ascending order
image_files.sort()

# iterate over images in the directory
for filename in image_files:
    #if filename.endswith('.jpg') or filename.endswith('.png'):
        
        #image_path = os.path.join(image_dir, filename)
    image = cv2.imread(filename)
        
       # resize image
    resized_image = resize_image(image)
        
    # extract features
    # 1x1
    color_hist_rgb, color_hist_hsv, color_hist_luv = extract_color_histograms(resized_image)
    # 画像2x2分割
    color_hist_rgb2, color_hist_hsv2, color_hist_luv2 = extract_color_histograms2(resized_image)
    # 3x3
    color_hist_rgb3, color_hist_hsv3, color_hist_luv3 = extract_color_histograms3(resized_image)

    gabor_features = extract_gabor_features(resized_image)
    dcnn_features = extract_dcnn_features(resized_image)
        
        # append features to lists
    all_features_rgb.append(color_hist_rgb)
    all_features_hsv.append(color_hist_hsv)
    all_features_luv.append(color_hist_luv)

    all_features_rgb2.append(color_hist_rgb2)
    all_features_hsv2.append(color_hist_hsv2)
    all_features_luv2.append(color_hist_luv2)
    all_features_rgb3.append(color_hist_rgb3)
    all_features_hsv3.append(color_hist_hsv3)
    all_features_luv3.append(color_hist_luv3)
    
    all_features_gabor.append(gabor_features)
    all_features_dcnn.append(dcnn_features)
    
    # break if 300 processed
    if len(all_features_rgb) >= 300:
        break

# convert to numpy arrays
all_features_rgb = np.array(all_features_rgb)
all_features_hsv = np.array(all_features_hsv)
all_features_luv = np.array(all_features_luv)

all_features_rgb2 = np.array(all_features_rgb2)
all_features_hsv2 = np.array(all_features_hsv2)
all_features_luv2 = np.array(all_features_luv2)
all_features_rgb3 = np.array(all_features_rgb3)
all_features_hsv3 = np.array(all_features_hsv3)
all_features_luv3 = np.array(all_features_luv3)

all_features_gabor = np.array(all_features_gabor)
all_features_dcnn = np.array(all_features_dcnn)

# save
np.save('all_features_rgb.npy', all_features_rgb)
np.save('all_features_hsv.npy', all_features_hsv)
np.save('all_features_luv.npy', all_features_luv)
np.save('all_features_rgb2.npy', all_features_rgb2)
np.save('all_features_hsv2.npy', all_features_hsv2)
np.save('all_features_luv2.npy', all_features_luv2)
np.save('all_features_rgb3.npy', all_features_rgb3)
np.save('all_features_hsv3.npy', all_features_hsv3)
np.save('all_features_luv3.npy', all_features_luv3)
np.save('all_features_gabor.npy', all_features_gabor)
np.save('all_features_dcnn.npy', all_features_dcnn)