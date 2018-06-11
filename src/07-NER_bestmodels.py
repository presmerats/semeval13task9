from NERmodel import NERmodel
import time
import gc
import pprint
import os



NERmodel.parallelbatchTraining('../data/batches/b0524/d24tDBmNERaCRFw1.yaml', status=False)
gc.collect()

NERmodel.parallelbatchTraining('../data/batches/b0524/d24tMDmNERaCRFw1.yaml', status=False)


NERmodel.parallelbatchTraining('../data/batches/b0524/d24tDBmNERaLSVMw3.yaml', status=False)


NERmodel.parallelbatchTraining('../data/batches/b0524/d24tMDmNERaLSVMw3.yaml', status=False)



