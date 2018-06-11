from NERmodel import NERmodel
from DDImodel import DDImodel
import time
import gc
import pprint
import os

NERmodel.parallelbatchCVTraining('../data/batches/cv/cv.yaml', status=False)
gc.collect()

# NERmodel.parallelbatchCVTraining('../data/batches/dnr22/d24tDBmNERaCRFw1m10.yaml', status=False)
# gc.collect()
# time.sleep(10)
# NERmodel.parallelbatchCVTraining('../data/batches/dnr22/d24tMDmNERaCRFw1m10.yaml', status=False)
# gc.collect()
# time.sleep(10)

# DDImodel.parallelbatchCVTraining('../data/batches/ddi22/ddi010.yaml', status=False)
# gc.collect()
# time.sleep(10)
# DDImodel.parallelbatchCVTraining('../data/batches/ddi22/ddi011.yaml', status=False)
# gc.collect()
# time.sleep(10)

## DDImodel.parallelbatchCVTraining('../data/batches/ddi22/ddi012.yaml', status=False)
## gc.collect()
## time.sleep(10)
## DDImodel.parallelbatchCVTraining('../data/batches/ddi22/ddi013.yaml', status=False)
## gc.collect()
## time.sleep(10)


# NERmodel.parallelbatchCVTraining('../data/batches/dnr22/d23tDBmBERaLSVMw3m0.yaml', status=False)
# gc.collect()
# time.sleep(10)
# NERmodel.parallelbatchCVTraining('../data/batches/dnr22/d23tMDmBERaLSVMw3m0.yaml', status=False)
# gc.collect()
# time.sleep(10)


