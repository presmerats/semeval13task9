from NERmodel import NERmodel
import time
import gc
import pprint
import os

NERmodel.parallelbatchTraining('../data/batches/b0520/d20tDBmNERaCRFw3.yaml', status=False)
gc.collect()
time.sleep(60)
