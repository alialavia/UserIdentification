#!/usr/bin/env python2

import os
import time

import numpy as np

import openface
import openface.helper
from openface.data import iterImgs

# path managing
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../..', 'models')	# path to the model directory
dlibModelDir = os.path.join(modelDir, 'dlib')		# dlib face detector model
openfaceModelDir = os.path.join(modelDir, 'openface')

# classifiers
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# argument container
# TODO: refactor this properly!
class Arguments:
    def __init__(self):
        self.dlibFacePredictor = "shape_predictor_68_face_landmarks.dat"
        self.landmarks = "outerEyesAndNose"
        self.size = 96
        self.skipMulti = True
        self.verbose = True
        # embedding calculation
        self.networkModel = os.path.join(openfaceModelDir,'nn4.small2.v1.t7') # torch network model
        self.cuda = False


class OfflineUserClassifier:
    # key: user id, value: list of embeddings
    user_embeddings = {}    # raw embeddings
    neural_net = None       # torch network
    dlib_aligner = None     # dlib face aligner
    classifier = None              # classifier
    label_encoder = None    # classifier label encoder

    def __init__(self):
        args = Arguments()

        start = time.time()
        print "--- loading models..."
        # load neural net
        self.neural_net = openface.TorchNeuralNet(args.networkModel, imgDim=args.size, cuda=args.cuda)
        # load dlib model
        self.dlib_aligner = openface.AlignDlib(dlibModelDir + "/" + args.dlibFacePredictor)
        # initialize classifier
        self.classifier = SVC(C=1, kernel='linear', probability=True)

        print("--- identifier initialization took {} seconds".format(time.time() - start))

    def collect_embeddings(self, images, user_id):
        """collect embeddings of faces to train detect - for a single user id"""

        args = Arguments()

        print "--- Starting normalization..."
        # normalize images
        images_normalized = []
        start = time.time()
        if len(images) > 0:
            for imgObject in images:
                # align face - ignore images with multiple bounding boxes
                aligned = self.align_face(imgObject, args.landmarks, args.size)
                if aligned is not None:
                    images_normalized.append(aligned)

        if len(images_normalized) > 0:
            print("--- Alignment took {} seconds - " + str(len(images_normalized)) + "/" + str(len(images)) + " images suitable".format(time.time() - start))

        else:
            print "--- No suitable images (no faces detected)"

        # generate embeddings
        reps = []
        for img in images_normalized:
            start = time.time()
            rep = self.neural_net.forward(img)
            print("--- = Neural network forward pass took {} seconds.".format(
                time.time() - start))
            reps.append(rep)

        # save
        if user_id in self.user_embeddings:
            # append
            print "--- Appending "+str(len(reps))+" embeddings"
            self.user_embeddings[user_id].append(reps)
        else:
            self.user_embeddings[user_id] = reps

        # display current representations
        self.print_embedding_status()

    def identify_user(self, user_img):

        start = time.time()
        embedding = self.get_embedding(user_img)

        if embedding is None:
            return None

        embedding = embedding.reshape(1, -1)

        user_id = self.classifier.predict(embedding)

        # prediction probabilities
        probabilities = self.classifier.predict_proba(embedding).ravel()
        maxI = np.argmax(probabilities)
        confidence = probabilities[maxI]
        user_id_pred = self.label_encoder.inverse_transform(maxI)

        print "========= USER ID: " + str(user_id) + " | max prob: " + str(user_id_pred)
        print("--- Identification took {} seconds.".format(time.time() - start))

        return user_id
    #  ----------- UTILITIES

    def get_embedding(self, user_img):
        args = Arguments()
        # align image
        normalized = self.align_face(user_img, args.landmarks, args.size)
        if normalized is None:
            return None

        # generate embedding
        rep = self.neural_net.forward(normalized)
        return rep

    def print_embedding_status(self):
        print "--- Current embeddings:"
        for user_id, embeddings in self.user_embeddings.iteritems():
            print "     User" + str(user_id) + ": " + str(len(embeddings)) + " representations"

    def test_training(self):

        start = time.time()
        # Matthias: nr 2
        labels = [ 1.  ,1.  ,2. ,2.]
        embeddings_accumulated = [[0.09432523,0.12247606,-0.12297968,0.02930461,0.04886347,0.08199037
                            ,0.06472751,-0.02508215,-0.02965139,-0.02833769,0.181373,0.1023135
                            ,0.06199017,-0.09017225,-0.18463612,-0.05030936,-0.08972715,0.06558438
                            ,-0.12173051,0.12092381,0.13721439,-0.09082257,-0.01291597,0.08284048
                            ,0.04908861,-0.08999868,0.04068245,0.01948535,0.08204082,0.16066657
                            ,-0.10866562,-0.0766836,-0.02778343,0.04692072,-0.09657263,-0.09822231
                            ,-0.13035585,-0.04184557,0.0096476,0.00440672,-0.01385642,-0.08501584
                            ,0.04308448,0.17249137,-0.12994982,-0.17400444,0.05614032,0.0046062
                            ,-0.05324974,0.10809524,0.0653071,0.10456908,-0.17619832,0.03161304
                            ,0.02034661,-0.00190462,-0.10708237,0.09455792,-0.04039523,-0.02815481
                            ,-0.06887836,-0.08682755,0.24910021,0.02310155,-0.08429448,-0.06831464
                            ,-0.03438343,-0.08429162,-0.09301203,0.03302766,0.00281743,0.04523946
                            ,0.02600059,0.02626732,0.01779462,-0.04145556,0.06151537,-0.01192219
                            ,-0.05321243,0.11259104,-0.01813241,-0.05334203,0.0941195,0.04767558
                            ,-0.04010718,0.10359764,0.01727163,-0.03103522,0.12276407,0.02795325
                            ,0.01482056,-0.20738073,-0.13220684,0.02609385,-0.03866709,-0.19990419
                            ,-0.10663626,0.01605269,0.02760329,-0.07138118,0.0086026,0.09569637
                            ,0.08187499,0.1027732,-0.04625086,0.12154666,0.01571439,0.23322205
                            ,0.02541953,-0.11369016,0.06821161,0.05144801,-0.06656482,0.13770872
                            ,-0.01340639,0.01532194,0.04323566,0.09173611,0.12800112,0.11954787
                            ,-0.03818309,-0.06452282,-0.05894236,0.00997317,-0.01616522,0.05427167
                            ,0.07759833,0.05407377]
                            ,[0.02534793,0.12704344,-0.06377918,-0.08964743,0.03958384,0.13560484
                            ,0.0035012,0.12131532,0.02049792,0.03225129,0.10027103,0.0632676
                            ,0.09628873,-0.26682347,-0.1487454,0.036475,-0.08827581,0.0474879
                            ,-0.24998708,0.14494638,0.1485538,-0.07132069,0.01243113,0.09340429
                            ,-0.10741381,-0.06797671,-0.00155025,-0.12232094,0.06687416,0.00206303
                            ,-0.01027272,-0.08739018,-0.01534874,0.09902387,-0.07582862,-0.06092755
                            ,-0.06440046,0.03969849,-0.08118446,0.05967991,-0.00958546,-0.12460437
                            ,-0.05070071,0.152931,-0.02723279,-0.1117252,0.08974932,0.03759511
                            ,-0.18740369,0.12804183,0.03153253,-0.05397858,-0.22248176,-0.01624606
                            ,-0.04928431,0.01752477,-0.09632535,0.0285293,-0.04041471,-0.06341214
                            ,-0.03793914,-0.05658839,0.17026722,-0.03033276,0.01987215,-0.24535324
                            ,-0.04265226,-0.00381506,-0.07393103,0.01991615,-0.01560289,-0.00946932
                            ,-0.028007,-0.02519011,-0.01282906,-0.01038657,-0.00338331,0.18679936
                            ,-0.01443517,0.05356421,-0.0701463,-0.03079787,0.00380396,0.07800649
                            ,-0.09848823,0.12533754,0.08338971,0.00735695,0.08807238,0.04192413
                            ,-0.01674767,-0.19535777,-0.15125977,0.04979046,-0.01501085,-0.03564126
                            ,-0.07011931,-0.023064,-0.10646579,-0.00225126,0.00540315,-0.00906458
                            ,-0.11427642,0.11107378,0.0187623,0.03234699,-0.06907182,0.15136175
                            ,-0.03762001,-0.09155462,-0.02706809,0.09671018,-0.03982377,0.07791182
                            ,0.00720286,0.03617542,-0.1255081,-0.00126609,0.01565578,0.12817368
                            ,-0.06263072,0.03989713,-0.0327239,0.10650044,0.01682332,-0.0047433
                            ,-0.05557182,-0.06834777]
                            ,[-0.03293964,-0.03750783,0.08285745,-0.02746028,0.13249229,0.07942843
                            ,-0.00185049,-0.0423231,-0.07338587,0.0513005,0.04307686,-0.04215237
                            ,0.02854649,0.02654963,0.19221313,0.06755589,0.00103392,0.15160589
                            ,0.04881257,-0.02956128,0.10941904,0.07206848,0.20571935,-0.04880722
                            ,0.06761299,0.0080359,-0.05090751,-0.13123247,0.14756088,-0.02483481
                            ,0.14339116,-0.00927146,0.07471134,-0.03000261,0.1765338,0.04990638
                            ,-0.0570038,0.0780702,0.01475807,-0.09579319,0.00527101,0.04473851
                            ,-0.06185326,0.15290466,-0.23528713,-0.02259201,0.01551868,0.04245473
                            ,-0.01091637,-0.0845984,-0.03155915,-0.03663119,-0.02265365,-0.0847896
                            ,-0.05572295,0.02460118,0.0844651,0.07954706,-0.04466062,0.02381799
                            ,-0.13665095,-0.08148899,0.08802734,-0.01846396,0.23154497,-0.01596716
                            ,-0.09807917,0.01349682,-0.05352049,0.15679178,0.01978051,0.1119616
                            ,-0.02404605,-0.10521396,-0.09569208,-0.10896765,-0.09421977,0.00090004
                            ,0.00550442,-0.0282906,0.01578465,-0.0008709,-0.00490644,0.01399914
                            ,-0.00314565,-0.16665979,0.03118968,0.06052634,-0.06618736,0.01639773
                            ,0.10232206,0.09464578,-0.11715083,-0.04553568,0.05986423,-0.05740212
                            ,-0.07977657,0.21588968,-0.07900454,0.03134699,0.02376053,0.01643077
                            ,0.04019921,-0.11844395,0.05300527,0.01761175,0.0993067,-0.08417895
                            ,0.34288689,-0.02931771,-0.05145922,0.04041028,-0.01789164,0.00431382
                            ,0.14387533,0.03182553,0.04244753,-0.06053276,-0.03500497,0.13936444
                            ,0.07459769,-0.07597688,-0.06877411,0.02591428,-0.04604036,0.03575793
                            ,0.10320232,0.11396465]
                            ,[-0.02454037,-0.04126243,0.01781121,0.02819706,0.1370606,0.0244129
                            ,-0.11071072,-0.09470159,0.02837539,-0.04899093,0.02654091,-0.0075549
                            ,0.03534986,0.00493263,0.11561734,0.09497358,0.07383484,0.01089744
                            ,0.07664657,0.07286123,0.14463024,-0.0013617,0.20021564,-0.02108853
                            ,0.12019048,0.04220056,-0.03664469,-0.17526518,0.09311989,0.01116637
                            ,0.22725081,0.06786175,0.1469795,-0.00175863,0.11865099,0.09986304
                            ,0.05186172,0.14269523,0.08285247,-0.1428991,0.00900867,0.02039703
                            ,-0.03871271,0.0344942,-0.12020127,0.01182013,0.05947608,0.02711469
                            ,0.04927597,-0.07964018,-0.10871163,0.0215562,0.09710088,-0.06542715
                            ,0.00732135,-0.0108269,0.1033585,-0.01869135,-0.08588462,-0.04791727
                            ,-0.11471377,-0.10092328,0.04747891,-0.07773249,0.15680663,-0.02630146
                            ,-0.04225538,0.08510342,0.00608907,0.08720449,-0.02552341,0.05696972
                            ,-0.15962642,-0.12662859,-0.00803069,-0.06892199,-0.04417069,0.06730556
                            ,-0.01042176,0.13118087,-0.07887538,0.05683753,0.01660625,0.04502281
                            ,0.07367159,-0.11838761,0.00794005,0.12619723,-0.06136678,-0.05289793
                            ,0.07901182,0.08821823,-0.04189871,0.06751327,0.01700043,-0.09384294
                            ,-0.05724768,0.20576344,-0.08219636,-0.0692063,0.01362741,-0.06777747
                            ,-0.04109306,-0.02518863,0.04794309,0.03031386,0.10079048,-0.04678678
                            ,0.30172178,-0.0102096,-0.14324649,0.01651655,-0.1016923,0.01536264
                            ,0.13030338,0.03949744,-0.06277226,-0.1305024,-0.0554248,0.1892141
                            ,0.05560551,0.09002611,-0.05896085,0.05265835,-0.06646401,0.02584325
                            ,0.12777153,0.11292588]]

        self.label_encoder = LabelEncoder().fit(labels)
        labelsNum = self.label_encoder.transform(labels)

        # train classifier
        self.classifier.fit(embeddings_accumulated, labelsNum)
        print("    Classifier training took {} seconds".format(time.time() - start))

    def trigger_training(self):
        """triggers the detector training from the collected faces"""
        print "--- Triggered classifier training"

        if len(self.user_embeddings) < 2:
            print "Number of users must be greater than one"
            return

        start = time.time()
        embeddings_accumulated = []
        labels = []
        for user_id, user_embeddings in self.user_embeddings.iteritems():
            # add label
            labels = np.append(labels, np.repeat(user_id, len(user_embeddings)))
            print "user embeddings: "+str(user_id)
            print(user_embeddings)
            if not embeddings_accumulated:
                embeddings_accumulated = user_embeddings
            else:
                embeddings_accumulated = np.concatenate((embeddings_accumulated, user_embeddings))

        print "labels: "
        print(labels)
        print "embeddings:"
        embeddings_accumulated = np.array(embeddings_accumulated)

       # print(embeddings_accumulated)

        self.label_encoder = LabelEncoder().fit(labels)
        labelsNum = self.label_encoder.transform(labels)

        # train classifier
        self.classifier.fit(embeddings_accumulated, labelsNum)
        #print("    Classifier training took %s seconds for "+str(len(embeddings_accumulated))+" embeddings." % time.time() - start)

    def align_face(self, image, landmark, output_size, skip_multi=False):

        landmarkMap = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if landmark not in landmarkMap:
            raise Exception("Landmarks unrecognized: {}".format(landmark))

        landmarkIndices = landmarkMap[landmark]

        # TODO: check if is really output size or input size
        # align image
        outRgb = self.dlib_aligner.align(output_size, image,
                             landmarkIndices=landmarkIndices,
                             skipMulti=skip_multi)
        if outRgb is None:
            print("--- Unable to align.")

        return outRgb