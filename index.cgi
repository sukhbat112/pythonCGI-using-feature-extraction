#!/usr/local/anaconda3/bin/python3

import os
import cgi
import cgitb
import numpy as np
from similarity import sort_images

cgitb.enable()

# directory symbolic link
image_dir = 'img_kadai3a'

# load features files
all_features_rgb = np.load('all_features_rgb.npy')
all_features_hsv = np.load('all_features_hsv.npy')
all_features_luv = np.load('all_features_luv.npy')
all_features_rgb2 = np.load('all_features_rgb2.npy')
all_features_hsv2 = np.load('all_features_hsv2.npy')
all_features_luv2 = np.load('all_features_luv2.npy')
all_features_rgb3 = np.load('all_features_rgb3.npy')
all_features_hsv3 = np.load('all_features_hsv3.npy')
all_features_luv3 = np.load('all_features_luv3.npy')

all_features_gabor = np.load('all_features_gabor.npy')
all_features_dcnn = np.load('all_features_dcnn.npy')


# drop-down list : feature type
def generate_feature_type_options(selected):
    options = ['rgb' ,'rgb 2x2' ,'rgb 3x3' ,'hsv', 'hsv 2x2', 'hsv 3x3', 'luv', 'luv 2x2', 'luv 3x3', 'gabor', 'dcnn']
    dropdown = '<select name="feature_type">'
    for option in options:
        if option == selected:
            dropdown += f'<option value="{option}" selected>{option.upper()}</option>'
        else:
            dropdown += f'<option value="{option}">{option.upper()}</option>'
    dropdown += '</select>'
    return dropdown

# drop-down list : distance metric
def generate_distance_metric_options(selected):
    options = ['euclidean', 'histogram_intersection']
    dropdown = '<select name="distance_metric">'
    for option in options:
        if option == selected:
            dropdown += f'<option value="{option}" selected>{option}</option>'
        else:
            dropdown += f'<option value="{option}">{option}</option>'
    dropdown += '</select>'
    return dropdown

# parse form inputs
form = cgi.FieldStorage()
query_image_index = int(form.getvalue('query_image_index', 230))
feature_type = form.getvalue('feature_type', 'rgb')
distance_metric = form.getvalue('distance_metric', 'euclidean')


# sort images
sorted_indices, distances = sort_images(query_image_index, feature_type, distance_metric)


# generate HTML
print("Content-type: text/html\n")
print("<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 3.2//EN\">")
print(f"<title>Search Results by {feature_type.upper()} with {distance_metric.replace('_', ' ').title()}</title>")
print(f"<h1>Image Search by {feature_type.upper()} with {distance_metric.replace('_', ' ').title()}</h1>")

print("<form method='get'>")
print(f"<label>Query Image Index (range: 0-299): <input type='text' name='query_image_index' value='{query_image_index}'></label><br>")
print(f"<label>Feature Type: {generate_feature_type_options(feature_type)}</label><br>")
print(f"<label>Distance Metric: {generate_distance_metric_options(distance_metric)}</label><br>")
print("<input type='submit' value='Submit'>")
print("</form>")

# display query ikmage
print("<h2>Query Image</h2>")
query_image_path = os.path.join(image_dir, f'w{query_image_index:04d}.jpg')
print("<table border=\"1\" cellspacing=\"1\" cellpadding=\"2\">")
print("<tbody><tr>")
print(f"<td align=\"CENTER\"><a href=\"{query_image_path}\"><img src=\"{query_image_path}\" width=\"150\" height=\"120\"></a><br>Image Index :  [{query_image_index}] <br> Query Image</td>")
print("</tr></tbody></table>")

# display most similar
print("<hr><h2>50 Most similar images</h2>")
print("<table border=\"1\" cellspacing=\"1\" cellpadding=\"2\">")
print("<tbody><tr>")
for i, index in enumerate(sorted_indices[:50], start=1):
    image_path = os.path.join(image_dir, f'w{index:04d}.jpg')
    print(f"<td align=\"CENTER\"><a href=\"{image_path}\"><img src=\"{image_path}\" width=\"120\" height=\"120\"></a><br>Index: [{index}] <br> Score: {distances[index]:.5f}</td>")
    if i % 10 == 0:
        print("</tr><tr>")
print("</tr></tbody></table>")

#display least similar images
print("<hr><h2>50 Least similar images</h2>")
print("<table border=\"1\" cellspacing=\"1\" cellpadding=\"2\">")
print("<tbody><tr>")
for i, index in enumerate(sorted_indices[-50:], start=1):
    image_path = os.path.join(image_dir, f'w{index:04d}.jpg')
    print(f"<td align=\"CENTER\"><a href=\"{image_path}\"><img src=\"{image_path}\" width=\"120\" height=\"120\"></a><br>Index: [{index}] <br> Score: {distances[index]:.5f}</td>")
    if i % 10 == 0:
        print("</tr><tr>")
print("</tr></tbody></table>")

print("<br>&nbsp;&nbsp;<br><br>")
print("</body></html>")
