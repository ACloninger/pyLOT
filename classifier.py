import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class LOTClassifier:
    @staticmethod
    def get_best_classifier(X_train, y_train, X_test, y_test):
        names = ["KNN", "Linear SVM", "RBF SVM"]
        classifiers = [
            KNeighborsClassifier(n_neighbors=3),
            SVC(kernel="linear"),
            SVC(kernel='rbf')
        ]

        best_classifier = None
        max_score = 0.0
        best_class_name = ''

        for name, clf in zip(names, classifiers):
            cv_accs = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
            score = 100 * np.mean(cv_accs)
            print(f'{name} Validation Avg. Accuracy: {score:.2f}, Std: {np.std(cv_accs):.2f}')

            clf.fit(X_train, y_train)
            print(f'Classifier = {name}, Test Accuracy = {100.0 * clf.score(X_test, y_test):.2f}')

            yhat_test = clf.predict(X_test)
            print(classification_report(y_test, yhat_test))

            if score > max_score:
                best_classifier = clf
                max_score = score
                best_class_name = name

        print('-' * 80)
        print(f'Best --> Classifier = {best_class_name}, Test Accuracy = {max_score:.2f}')

        return best_classifier