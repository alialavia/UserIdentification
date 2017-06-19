from sklearn.model_selection import KFold
import logging
from sklearn.datasets import fetch_lfw_people

from uids.online_learning.BinaryThreshold import BinaryThreshold





def test_lfw():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    ###############################################################################
    # Download the data, if not already on disk and load it as numpy arrays

    lfw_people = fetch_lfw_people(min_faces_per_person=150, resize=0.4)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)


    ###############################################################################
    # Split into a training set and a test set using a stratified k fold


    # KFold(n_splits=2, random_state=None, shuffle=False)
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)
    print kf


    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)

        t = BinaryThreshold()
        t.partial_fit(train)

        pred_thresh = t.predict(ul, True)
        print pred_thresh
        print "Misdetections Thresholding (ul): {}".format(len(pred_thresh[pred_thresh > 0]))

        cv_clf.fit(X_train[train], y_train[train])

def test_lfw_pairwise():

    pass


# ================================= #
#              Main

if __name__ == '__main__':
    test_lfw_pairwise()