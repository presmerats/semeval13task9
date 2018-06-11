from DDImodel import DDImodel
import time
import gc
import pprint
import os


# trigrams and wordcounts as features
DDImodel.featureExtractionDDIstep2('../data/batches/ddi22/featureExtractionTest2type.yaml',status=False)
shutil.copyfile('debug_offsets.json','debug_offsetsBDDI_test.json')

