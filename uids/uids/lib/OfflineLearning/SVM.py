#!/usr/bin/env python2
# specific classifier
from sklearn.svm import SVC as sk_SVM
# classifier superclass
from uids.lib.OfflineLearning.OfflineClassifierBase import OfflineClassifierBase


class SVM(OfflineClassifierBase):

    def __init__(self, user_db_):
        OfflineClassifierBase.__init__(self, user_db_)    # superclass init

    def define_classifier(self):
        self.classifier_tag = 'Linear_SVM'
        self.classifier = sk_SVM(C=1, kernel='linear', probability=True)

    # --------------------------------

    # def ___identify_user_prob(self, user_img):
    #
    #     if self.training_status is False:
    #         return (None, None, None)
    #
    #     start = time.time()
    #     embedding = self.get_embedding(user_img)
    #
    #     if embedding is None:
    #         return (None, None, None)
    #
    #     embedding = embedding.reshape(1, -1)
    #
    #     # alternative - predicts index of label array
    #     # user_id = self.classifier.predict(embedding)
    #
    #     # y_pred = clf.predict(X)
    #
    #     # prediction probabilities
    #     probabilities = self.classifier.predict_proba(embedding).ravel()
    #     maxI = np.argmax(probabilities)
    #     confidence = probabilities[maxI]
    #     user_id_pred = self.label_encoder.inverse_transform(maxI)
    #
    #
    #     nice_name = self.user_list[int(user_id_pred)]
    #
    #     if np.shape(probabilities)[0] > 1:
    #         for i, prob in enumerate(probabilities):
    #             # label encoder id: np.int64()
    #             label = self.label_encoder.inverse_transform(np.int64(i))
    #             print "    label: "+ str(label) + " | prob: " + str(prob)
    #
    #     print("--- Identification took {} seconds.".format(time.time() - start))
    #     return (user_id_pred, nice_name, confidence)

