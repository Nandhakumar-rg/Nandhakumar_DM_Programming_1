# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, cross_validate, train_test_split
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        #loading the MNIST dataset
        mnist = fetch_openml('mnist_784')
        X, y = mnist.data, mnist.target
        
        # Split X and y into train and test sets
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
        
        answer = {}
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary


        
        # Print out the number of elements in each class y
        unique_train, counts_train = np.unique(ytrain, return_counts=True)
        unique_test, counts_test = np.unique(ytest, return_counts=True)

        # Print out the number of classes for both training and testing datasets
        num_classes_train = len(unique_train)
        num_classes_test = len(unique_test)

        # Print out the number of elements in each class in the training and testing set
        class_count_train = np.bincount(ytrain)
        class_count_test = np.bincount(ytest)

        # Print out the length of Xtrain, Xtest, ytrain, ytest
        length_Xtrain = len(Xtrain)
        length_Xtest = len(Xtest)
        length_ytrain = len(ytrain)
        length_ytest = len(ytest)

        # Print out the maximum value in the training and testing set
        max_Xtrain = Xtrain.max()
        max_Xtest = Xtest.max()

        # Fill the answer dictionary
        answer["nb_classes_train"] = num_classes_train
        answer["nb_classes_test"] = num_classes_test
        answer["class_count_train"] = class_count_train
        answer["class_count_test"] = class_count_test
        answer["length_Xtrain"] = length_Xtrain
        answer["length_Xtest"] = length_Xtest
        answer["length_ytrain"] = length_ytrain
        answer["length_ytest"] = length_ytest
        answer["max_Xtrain"] = max_Xtrain
        answer["max_Xtest"] = max_Xtest

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        
        train_sizes = [1000, 5000, 10000]
        test_sizes = [200, 1000, 2000]
        answers = {}



        for ntrain in train_sizes:
            answers[ntrain] = {}
            for ntest in test_sizes:
                    Xtrain, ytrain = X[:ntrain], y[:ntrain]
                    Xtest, ytest = X[ntrain:ntrain+ntest], y[ntrain:ntrain+ntest]


                    

                    ## PART C
                    clf_c = DecisionTreeClassifier(random_state=self.seed)
                    cv_c = KFold(n_splits=5, shuffle=True, random_state=self.seed)
                    cv_results = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=clf_c, cv=cv_c)
                    answer_dt_c = {}
                    answer_dt_c['clf'] = clf_c
                    answer_dt_c['cv'] = cv_c
                    scores_dt_c = {}


                    mean_accuracy = cv_results['test_score'].mean()
                    std_accuracy = cv_results['test_score'].std()

                    mean_fit_time = cv_results['fit_time'].mean()
                    std_fit_time = cv_results['fit_time'].std()

                    scores_dt_c['mean_fit_time'] = mean_fit_time
                    scores_dt_c['std_fit_time'] = std_fit_time
                    scores_dt_c['mean_accuracy'] = mean_accuracy
                    scores_dt_c['std_accuracy'] = std_accuracy

                    answer_dt_c['scores_C'] = scores_dt_c

                    ## PART D
                    clf_d = DecisionTreeClassifier(random_state=self.seed)
                    cv_d = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)
                    cv_results_d = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=clf_d, cv=cv_d)

                    answer_dt_d = {}

                    answer_dt_d['clf'] = clf_d
                    answer_dt_d['cv'] = cv_d

                    scores_d = {}



                    mean_accuracy = cv_results_d['test_score'].mean()
                    std_accuracy = cv_results_d['test_score'].std()

                    mean_fit_time = cv_results_d['fit_time'].mean()
                    std_fit_time = cv_results_d['fit_time'].std()

                    scores_d['mean_fit_time'] = mean_fit_time
                    scores_d['std_fit_time'] = std_fit_time
                    scores_d['mean_accuracy'] = mean_accuracy
                    scores_d['std_accuracy'] = std_accuracy

                    answer_dt_d['scores_D'] = scores_d
                    answer_dt_d['explain_kfold_vs_shuffle_split'] = """
                        K-Fold Cross-Validation splits the dataset into k consecutive folds, each fold used once as a test set while the k-1 remaining folds form the training set. It ensures that every observation from the original dataset has the chance of appearing in training and test set. It's well-suited for smaller datasets where maximizing the amount of training data is critical.

                        Shuffle-Split Cross-Validation generates a user-defined number of independent train/test dataset splits. Samples are first shuffled and then split into a pair of train and test sets. It's more flexible than k-fold, allowing for control over the number of iterations and the size of the test sets. It can be more suitable for large datasets or when requiring more randomness in the selection of train/test samples.

                        Pros of Shuffle-Split:
                        - More control over the size of the test set and the number of resampling iterations.
                        - Better for large datasets due to its randomness and efficiency.

                        Cons of Shuffle-Split:
                        - Less systematic coverage of the data compared to k-fold.
                        - Potential for higher variance in test performance across iterations due to randomness.

                        Empirical advantages:
                        - Takes less time
                        - More accuracy
                    """
            ##Part F
            partF = {}
            clf_F = LogisticRegression(max_iter=300,random_state=self.seed)
            cv_F = ShuffleSplit(n_splits=5,random_state=self.seed)
            scores_trainlr = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=LogisticRegression(max_iter=300,random_state=self.seed), cv=ShuffleSplit(n_splits=5,random_state=self.seed))
                       
                    
            clf_F.fit(Xtrain,ytrain)
            y_train_pred = clf_F.predict(Xtrain)
            y_test_pred = clf_F.predict(Xtest)
            partF["clf"] =  clf_F
            confusion_matrix_train = confusion_matrix(ytrain, y_train_pred)
            confusion_matrix_test = confusion_matrix(ytest,y_test_pred)
            partF["cv"] = cv_F
            partF["scores_train_F"] = clf_F.score(Xtrain,ytrain)
            partF["scores_test_F"] = clf_F.score(Xtest,ytest)
            partF["mean_cv_accuracy_F"] = np.mean(scores_trainlr['test_score']) 
            partF["conf_mat_train"] = confusion_matrix_train
            partF["conf_mat_test"] = confusion_matrix_test

            answers[ntrain][ntest] = {
                        "partC": answer_dt_c,
                        "partD": answer_dt_d,
                        "partF": partF,
                        "ntrain": ntrain,
                        "ntest": ntest,
                        "class_count_train": list(np.bincount(ytrain)),
                        "class_count_test": list(np.bincount(ytest))
                    }
                # Enter your code and fill the `answer`` dictionary
            return answers
            
"""
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """
            

