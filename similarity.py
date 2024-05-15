import numpy as np


# load features from files
all_features_rgb = np.load('all_features_rgb.npy')
all_features_hsv = np.load('all_features_hsv.npy')
all_features_luv = np.load('all_features_luv.npy')
all_features_gabor = np.load('all_features_gabor.npy')
all_features_dcnn = np.load('all_features_dcnn.npy')
# 2x2 features
all_features_rgb2 = np.load('all_features_rgb2.npy')  
all_features_hsv2 = np.load('all_features_hsv2.npy')
all_features_luv2 = np.load('all_features_luv2.npy')
# 3x3 features
all_features_rgb3 = np.load('all_features_rgb3.npy')  
all_features_hsv3 = np.load('all_features_hsv3.npy')
all_features_luv3 = np.load('all_features_luv3.npy')


# Euclidean distance between two feature vectors
def euclidean_distance(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

# histogram intersection
def histogram_intersection(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))

def sort_images(query_image_index, feature_type, distance_metric):

    #辞書：特徴の種類とそのファイル
    feature_dict = {
        'rgb': all_features_rgb,
        'hsv': all_features_hsv,
        'luv': all_features_luv,
        'gabor': all_features_gabor,
        'dcnn': all_features_dcnn,

        'rgb 2x2': all_features_rgb2,
        'hsv 2x2': all_features_hsv2,
        'luv 2x2': all_features_luv2,
        'rgb 3x3': all_features_rgb3,
        'hsv 3x3': all_features_hsv3,
        'luv 3x3': all_features_luv3
    }

    if feature_type not in feature_dict:
        raise ValueError("Invalid feature type")

    query_features = feature_dict[feature_type][query_image_index]
    image_features = feature_dict[feature_type]

    # Calculate distances and sort images
    distances = []
    for feature in image_features:
        if distance_metric == 'euclidean':
            dist = euclidean_distance(query_features, feature)
        elif distance_metric == 'histogram_intersection':
            dist = histogram_intersection(query_features, feature)
        else:
            raise ValueError("distance metric error")
        distances.append(dist)
    
    # sort images by similarity
    sorted_indices = np.argsort(distances)

    if distance_metric== 'histogram_intersection':
        sorted_indices = sorted_indices[::-1]  # sort in descending order if histogram intersection
    
    return sorted_indices, distances