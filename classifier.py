import numpy as np
from sklearn.model_selection import cross_val_score  # For cross-validation
from sklearn.metrics import classification_report  # For detailed classification metrics
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn.svm import SVC  # Support Vector Classifier

class LOTClassifier:
    @staticmethod
    def get_best_classifier(X_train, y_train, X_test, y_test):
        """
        Determines the best classifier based on cross-validated accuracy from a set of classifiers.

        Parameters
        ----------
        X_train : np.array
            Training data features.
        
        y_train : np.array
            Training data labels.
        
        X_test : np.array
            Test data features.
        
        y_test : np.array
            Test data labels.

        Returns
        -------
        best_classifier : sklearn classifier
            The classifier with the highest cross-validated accuracy on the training set.
        """
        
        # List of classifier names
        names = ["KNN", "Linear SVM", "RBF SVM"]
        
        # List of classifier instances
        classifiers = [
            KNeighborsClassifier(n_neighbors=3),  # K-Nearest Neighbors with 3 neighbors
            SVC(kernel="linear"),  # Support Vector Classifier with a linear kernel
            SVC(kernel='rbf')  # Support Vector Classifier with an RBF (Gaussian) kernel
        ]

        # Initialize variables to track the best classifier
        best_classifier = None  # Will store the best classifier found
        max_score = 0.0  # Maximum cross-validated accuracy score
        best_class_name = ''  # Name of the best classifier

        # Loop through each classifier and evaluate its performance
        for name, clf in zip(names, classifiers):
            # Perform 5-fold cross-validation on the training set
            cv_accs = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
            
            # Calculate the mean accuracy across the 5 folds
            score = 100 * np.mean(cv_accs)
            
            # Print the validation results for the current classifier
            print(f'{name} Validation Avg. Accuracy: {score:.2f}%, Std: {np.std(cv_accs):.2f}')
            
            # Fit the classifier to the entire training set
            clf.fit(X_train, y_train)
            
            # Evaluate the classifier on the test set and print the accuracy
            print(f'Classifier = {name}, Test Accuracy = {100.0 * clf.score(X_test, y_test):.2f}%')

            # Predict the labels for the test set
            yhat_test = clf.predict(X_test)
            
            # Print a detailed classification report
            print(classification_report(y_test, yhat_test))

            # Update the best classifier if the current one has a higher cross-validated accuracy
            if score > max_score:
                best_classifier = clf
                max_score = score
                best_class_name = name

        # Print the name and test accuracy of the best classifier found
        print('-' * 80)
        print(f'Best --> Classifier = {best_class_name}, Test Accuracy = {max_score:.2f}%')

        # Return the best classifier
        return best_classifier
