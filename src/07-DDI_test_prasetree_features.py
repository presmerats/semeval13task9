from DDImodel import DDImodel
import time
import gc
import pprint
import os
import shutil


# # trigrams and words lists extraction and saving
# DDImodel.featureExtractionDDITest('../data/batches/ddi22/ddi001.yaml',status=False, treetype="Parse")

# trigrams and wordcounts as features
DDImodel.featureExtractionDDIstep2('../data/batches/ddi22/featureExtractionTest2type.yaml',status=False)
shutil.copyfile('debug_offsets.json','debug_offsetsBDDI_test.json')



DDImodel.featureExtractionDDIstep2('../data/batches/ddi22/featureExtractionTest2.yaml',status=False)
shutil.copyfile('debug_offsets.json','debug_offsetsBDDI_test.json')


