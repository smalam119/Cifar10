import sys
import numpy as np

class NaiveClassifier:

    def __init__(self, training_images, training_label, feature_extractor):
        self.training_images = training_images
        self.training_label = training_label
        self.feature_extractor = feature_extractor

    def extract_feature_from_single_image(self):
        return

    def extract_feature_from_multiple_images(self):
        return

    def classify_single_image(self, test_image):
        smallest_distance = sys.maxsize
        for x in range(len(self.training_images)):
            dist = self.calculate_euclidean_distance(self, test_image, self.training_images[x])
            if dist < smallest_distance:
                smallest_distance = dist
                predicted_label = self.training_label[x]
            if smallest_distance == 0:
                break
            else:
                continue
            break
        return predicted_label

    def classify_multiple_images(self, test_images):
        smallest_distance = sys.maxsize
        predicted_label_list = []
        for x in range(len(test_images)):
            for y in range(len(self.training_images)):
                dist = self.calculate_euclidean_distance(self, test_images[x], self.training_images[y])
                if dist < smallest_distance:
                    smallest_distance = dist
                    predicted_label = self.training_label[y]
                    predicted_label_list.append(predicted_label)
                if smallest_distance == 0:
                    break
                else:
                    continue
                break
        return predicted_label_list

    def calculate_euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
