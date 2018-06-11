from DDImodel import DDImodel
import time
import gc
import pprint
import os
import shutil
from FeatureExtractionDDI import FeatureExtractionDDI



# --Preprocessing-&-FeatureExtractionn-----------------------------

# Preprocessing
# trigrams and words lists extraction and saving
# DDImodel.featureExtractionDDI('../data/batches/ddi22/ddi001_MedLine.yaml', status=False)
# DDImodel.featureExtractionDDITest('../data/batches/ddi22/ddi001_MedLine.yaml',status=False)


# Add word and window features to .data
self = FeatureExtractionDDI('')
# self.wordFeaturesExtension(
#     filepath='../data/features/preprocessDDIMedline-preprocessDDIMedlinem0.json',
#     resultpath='../data/features/preprocessDDIMedline-step2.json',
#     filepath2='../data/features/preprocessDDIMedline-test.json',
#     resultpath2='../data/features/preprocessDDIMedline-test-step2.json')



# FeatureExtraction
# trigrams and wordcounts as features
# create the final training and test feature files

DDImodel.featureExtractionDDIstep2(
    '../data/batches/ddi22/preprocessBDDIstep3Medline.yaml',
    status=False)
shutil.copyfile('debug_offsets.json','debug_offsets_trainBDDI_MD.json')


DDImodel.featureExtractionDDIstep2(
    '../data/batches/ddi22/preprocessDDIstep3Medline.yaml',
    status=False)
shutil.copyfile('debug_offsets.json','debug_offsets_train_MD.json')


DDImodel.featureExtractionDDIstep2(
    '../data/batches/ddi22/preprocessBDDIstep3Medline-test.yaml',
    status=False)
shutil.copyfile('debug_offsets.json','debug_offsetsBDDI_test_MD.json')


DDImodel.featureExtractionDDIstep2(
    '../data/batches/ddi22/preprocessDDIstep3Medline-test.yaml',
    status=False)
shutil.copyfile('debug_offsets.json','debug_offsetsDDI_test_MD.json')


# --Training--------------------------------------------------------
# DDImodel.parallelDDIbatchTraining('../data/batches/ddi22/ddi006.yaml',status=False)

# DDImodel.parallelDDIbatchTraining('../data/batches/ddi22/ddi007.yaml',status=False)