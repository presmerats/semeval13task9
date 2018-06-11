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
# DDImodel.featureExtractionDDI('../data/batches/ddi22/ddi001.yaml', status=False)
# DDImodel.featureExtractionDDITest('../data/batches/ddi22/ddi001.yaml',status=False)


# Add word and window features to .data
self = FeatureExtractionDDI('')
# self.wordFeaturesExtension(
#     filepath='../data/features/preprocessDDIDrugBank-preprocessDDIMedlinem0.json',
#     resultpath='../data/features/preprocessDDIDrugBank-step2.json',
#     filepath2='../data/features/preprocessDDIDrugBank-test.json',
#     resultpath2='../data/features/preprocessDDIDrugBank-test-step2.json')



# FeatureExtraction
# trigrams and wordcounts as features
# create the final training and test feature files


# real work

DDImodel.featureExtractionDDIstep2(
    '../data/batches/ddi22/preprocessBDDIstep3DrugBank.yaml',
    status=False)
shutil.copyfile('debug_offsets.json','debug_offsets_trainBDDI_DB.json')


DDImodel.featureExtractionDDIstep2(
    '../data/batches/ddi22/preprocessDDIstep3DrugBank.yaml',
    status=False)
shutil.copyfile('debug_offsets.json','debug_offsets_train_DB.json')


DDImodel.featureExtractionDDIstep2(
    '../data/batches/ddi22/preprocessBDDIstep3DrugBank-test.yaml',
    status=False)
shutil.copyfile('debug_offsets.json','debug_offsetsBDDI_test_DB.json')


DDImodel.featureExtractionDDIstep2(
    '../data/batches/ddi22/preprocessDDIstep3DrugBank-test.yaml',
    status=False)
shutil.copyfile('debug_offsets.json','debug_offsetsDDI_test_DB.json')


# --Training--------------------------------------------------------
# DrugBank DDI, BDDI
DDImodel.parallelDDIbatchTraining('../data/batches/ddi22/ddi006.yaml',status=False)


# MedLine DDI, BDDI
DDImodel.parallelDDIbatchTraining('../data/batches/ddi22/ddi007.yaml',status=False)



