from DDImodel import DDImodel
import time
import gc
import pprint
import os
import shutil

# FeatureExtraction
# trigrams and wordcounts as features
DDImodel.parallelDDIbatchTraining('../data/batches/ddi22/ddi008.yaml',status=False)