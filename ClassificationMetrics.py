class ClassificationMetrics:

    def __init__(self, training_labels, predicted_labels, predicted_scores):
        self.training_labels = training_labels
        self.predicted_labels = predicted_labels
        self.predicted_scores = predicted_scores

    def get_confusion_matrix_for_heat_map(self):
        return

    def calculate_accuracy(self):
        return

    def calculate_precision(self, confusion_matrix):
        true_positive = confusion_matrix[0][0]
        false_positive = confusion_matrix[1][0]
        precision = true_positive / (true_positive + false_positive)
        return precision

    def calculate_recall(self, confusion_matrix):
        true_positive = confusion_matrix[0][0]
        false_negative = confusion_matrix[0][1]
        recall = true_positive / (true_positive + false_negative)
        return recall

    def calculate_f1(self, confusion_matrix):
        precision = self.calculate_precision(confusion_matrix)
        recall = self.get_recall(confusion_matrix)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def calculate_roc_values(self):
        return

    def calculate_lift_values(self):
        return

