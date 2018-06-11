from NERmodel import NERmodel
import time
import gc
import pprint
import os


#NERmodel.parallelbatchTraining('../data/batches/combinationsER20180517.yaml', status=False)

#NERmodel.parallelbatchTraining('../data/batches/combinationsNER20180517.yaml', status=False)


#NERmodel.parallelbatchTraining('../data/batches/debug_crf00.yaml', status=False)


#NERmodel.parallelbatchTraining('../data/batches/debug_crf01.yaml', status=False)
#time.sleep(60)

#NERmodel.parallelbatchTraining('../data/batches/d20tMXmNWEaCRF.yaml', status=False)

#NERmodel.parallelbatchTraining('../data/batches/debug_window5.yaml', status=False)


##---linear svm 20180522----------------------------------------------------

# how long 4000, 6000, 10000, samples take for linear svm
#NERmodel.parallelbatchTraining('../data/batches/svmdebug/d21tDBmDDIaSVMw3.yaml', status=False)


# how long a full linear svm takes
#NERmodel.parallelbatchTraining('../data/batches/svmdebug/d23tDBmNERaSVMw3.yaml', status=False)

# linear weighted svm test
# NERmodel.parallelbatchTesting('../data/batches/svmdebug/d23tDBmNERaLWSVMw3.yaml', status=False, modelfile='../data/models/svm-pipeline_20180523090708.pkl')
# exit()
NERmodel.parallelbatchTraining('../data/batches/svmdebug/d23tDBmNERaLWSVMw3.yaml', status=False)
exit()


# --20180521-------------------------------------------------------------

# NERmodel.parallelbatchTraining('../data/batches/d20tMDmNERaSVMw5.yaml', status=False)
# gc.collect()
# time.sleep(60)
# exit()


for root, dirs, files in os.walk('../data/batches/b0522/', topdown = False):
    for name in files:
        if os.path.basename(name).endswith(".yaml"):
            try:
                NERmodel.parallelbatchTraining(os.path.join(root, name), status=True)
                gc.collect()
                time.sleep(30)
            except:
                pass


# done
NERmodel.parallelbatchTraining('../data/batches/b0520/d20tMDmNERaCRFw3.yaml', status=False)
gc.collect()
time.sleep(60)


NERmodel.parallelbatchTraining('../data/batches/b0520/d20tMDmNERaCRFw5.yaml', status=False)
gc.collect()
time.sleep(60)



NERmodel.parallelbatchTraining('../data/batches/b0520/d20tMDmNERaCRFw3.yaml', status=False)
gc.collect()
time.sleep(60)




NERmodel.parallelbatchTraining('../data/batches/b0520/d20tMXmNERaCRFw5.yaml', status=False)
gc.collect()
time.sleep(60)




NERmodel.parallelbatchTraining('../data/batches/b0520/d20tDBmNERaCRFw3.yaml', status=False)
gc.collect()
time.sleep(60)



NERmodel.parallelbatchTraining('../data/batches/b0520/d20tDBmNERaCRFw5.yaml', status=False)
gc.collect()
time.sleep(60)





# NERmodel.parallelbatchTraining('../data/batches/d20tMDmERaCRFw3.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tMDmERaCRFw5b.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tMXmERaCRFw3.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tMXmERaCRFw5.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tDBmERaCRFw3.yaml', status=False)
# gc.collect()
# time.sleep(60)

# NERmodel.parallelbatchTraining('../data/batches/d20tDBmERaCRFw5.yaml', status=False)
# gc.collect()
time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tMDmERaSVMw3.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tMXmERaSVMw3.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tDBmERaSVMw3.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tMDmNERaSVMw5.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tMXmNERaSVMw5.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tDBmNERaSVMw5.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tMDmERaSVMw5.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tMXmERaSVMw5.yaml', status=False)
# gc.collect()
# time.sleep(60)




# NERmodel.parallelbatchTraining('../data/batches/d20tDBmERaSVMw5.yaml', status=False)
# gc.collect()
# time.sleep(60)






