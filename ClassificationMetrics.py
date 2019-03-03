import numpy as np


class ClassificationMetrics:

    confusion_matrix = np.zeros((2, 2), np.int32)

    def __init__(self, label, training_labels, predicted_labels, predicted_scores, file_path):
        self.training_labels = training_labels
        self.predicted_labels = predicted_labels
        self.predicted_scores = predicted_scores

        for x in range(len(training_labels)):
            if training_labels[x] == label and predicted_labels[x] == label:
                self.confusion_matrix[0][0] = self.confusion_matrix[0][0] + 1
            elif training_labels[x] == label and predicted_labels[x] != label:
                self.confusion_matrix[1][0] = self.confusion_matrix[1][0] + 1
            elif training_labels[x] != label and predicted_labels[x] == label:
                self.confusion_matrix[0][1] = self.confusion_matrix[0][1] + 1
            elif training_labels[x] != label and predicted_labels[x] != label:
                self.confusion_matrix[1][1] = self.confusion_matrix[1][1] + 1

        np.savetxt(file_path + "/confusion_matrix.txt", self.confusion_matrix)


    def get_confusion_matrix_for_heat_map(self):
        return

    def calculate_accuracy(self):
        true_positive = self.confusion_matrix[0][0]
        true_negative = self.confusion_matrix[1][1]
        total = self.confusion_matrix[0][0] + self.confusion_matrix[0][1] + self.confusion_matrix[1][0] + self.confusion_matrix[1][1]
        accuracy = (true_negative + true_positive) / total
        return accuracy

    def calculate_precision(self):
        true_positive = self.confusion_matrix[0][0]
        false_positive = self.confusion_matrix[1][0]
        precision = true_positive / (true_positive + false_positive)
        return precision

    def calculate_recall(self):
        true_positive = self.confusion_matrix[0][0]
        false_negative = self.confusion_matrix[0][1]
        recall = true_positive / (true_positive + false_negative)
        return recall

    def calculate_f1(self):
        precision = self.calculate_precision(self.confusion_matrix)
        recall = self.get_recall(self.confusion_matrix)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def get_specificity(self, confusion_matrix):
        true_negative = confusion_matrix[1][1]
        false_positive = self.confusion_matrix[1][0]
        specificity = true_negative / (true_negative + false_positive)
        return specificity

    def calculate_roc_values(self, thresholds):
        return

    def calculate_lift_values(self):
        return

