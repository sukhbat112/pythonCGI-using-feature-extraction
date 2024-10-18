Image searching mini CGI project.

The files are just raw codes web server and datasets are excluded.

Extracts RGB, HSV, LUV color histogram, and Gabor(Grayscale), DCNN(VGG16) features from the image dataset and saves it to .npy file.
Then uses it to calculate the similarity between the query image and the rest of the database.
Similarity is calculated by two methods, histogram intersection and euclidian distance.
CGI script shows the search results after sorting the images based on the similarity.


Image is an example of HSV 2x2 (image divided into 4 parts and the histogram is calculated seperately and concatenated after)


![img_srch](https://github.com/sukhbat112/pythonCGI-using-feature-extraction/assets/68054312/7cee9fa9-c242-4d17-8b3b-cf324cf0b05e)
