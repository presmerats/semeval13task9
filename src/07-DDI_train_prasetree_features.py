from DDImodel import DDImodel
import time
import gc
import pprint
import os
import shutil


# # trigrams and words lists extraction and saving
# DDImodel.featureExtractionDDI('../data/batches/ddi22/ddi001.yaml', status=False, treetype="Dependency")

# # trigrams and words lists extraction and saving
# DDImodel.featureExtractionDDITest('../data/batches/ddi22/ddi001.yaml',status=False, treetype="Parse")


# ---- 32% F1 score----------------------------
# trigrams and wordcounts as features
# DDImodel.featureExtractionDDIstep2('../data/batches/ddi22/featureExtraction2.yaml',status=False)

# shutil.copyfile('debug_offsets.json','debug_offsets_train.json')

# # trigrams and wordcounts as features
# DDImodel.featureExtractionDDIstep2('../data/batches/ddi22/featureExtractionTest2.yaml',status=False)

# shutil.copyfile('debug_offsets.json','debug_offsets_test.json')

# DDImodel.parallelDDIbatchTraining('../data/batches/ddi22/ddi004.yaml',status=False)

# --- manual testing DDI vs BDDI-------------



# --Preprocessing-&-FeatureExtractionn-----------------------------

# Preprocessing
# # trigrams and words lists extraction and saving
# DDImodel.featureExtractionDDI('../data/batches/ddi22/ddi001.yaml', status=False, treetype="Dependency")
# DDImodel.featureExtractionDDITest('../data/batches/ddi22/ddi001.yaml',status=False, treetype="Parse")


# Add word and window features to .data
# self = FeatureExtractionDDI('')
# self.wordFeaturesExtension()


# FeatureExtraction
# trigrams and wordcounts as features
# create the final training and test feature files


# DDImodel.featureExtractionDDIstep2('../data/batches/ddi22/featureExtraction2.yaml',status=False)
# shutil.copyfile('debug_offsets.json','debug_offsets_trainBDDI.json')


# DDImodel.featureExtractionDDIstep2('../data/batches/ddi22/featureExtraction2type.yaml',status=False)
# shutil.copyfile('debug_offsets.json','debug_offsets_train.json')


# DDImodel.featureExtractionDDIstep2('../data/batches/ddi22/featureExtractionTest2type.yaml',status=False)
# shutil.copyfile('debug_offsets.json','debug_offsetsBDDI_test.json')



# DDImodel.featureExtractionDDIstep2('../data/batches/ddi22/featureExtractionTest2.yaml',status=False)
# shutil.copyfile('debug_offsets.json','debug_offsetsBDDI_test.json')


# --Training--------------------------------------------------------
# DDImodel.parallelDDIbatchTraining('../data/batches/ddi22/ddi006.yaml',status=False)

DDImodel.parallelDDIbatchTraining('../data/batches/ddi22/ddi007.yaml',status=False)