import threading
import time
import vart
import xir
import numpy


N_IMG = 1000


def get_DPU_iName(dpu):
    return dpu.get_input_tensors()[0].name


def get_DPU_oName(dpu):
    return dpu.get_output_tensors()[0].name


def execute_async(dpu, tensor_buffers_dict):
    input_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()
    ]
    output_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()
    ]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)


def ReLU(x):
    return x * (x > 0)


def Argmax(data):
    '''
    returns index of highest value in data
    '''
    val = numpy.argmax(data)
    return val


def LogSoftmax(x):
    c = x.max()
    logsumexp = numpy.log(numpy.exp(x - c).sum())
    return x - c - logsumexp


def runDPU(dpu_1, dpu_3, dpu_5, dpu_7, img):
    
    iTensor_1 = dpu_1.get_input_tensors()
    oTensor_1 = dpu_1.get_output_tensors()
    iTensor_3 = dpu_3.get_input_tensors()
    oTensor_3 = dpu_3.get_output_tensors()
    iTensor_5 = dpu_5.get_input_tensors()
    oTensor_5 = dpu_5.get_output_tensors()
    iTensor_7 = dpu_7.get_input_tensors()
    oTensor_7 = dpu_7.get_output_tensors()

    i1_ndim = tuple(iTensor_1[0].dims)
    i3_ndim = tuple(iTensor_3[0].dims)
    i5_ndim = tuple(iTensor_5[0].dims)
    i7_ndim = tuple(iTensor_7[0].dims)
    o1_ndim = tuple(oTensor_1[0].dims)
    o3_ndim = tuple(oTensor_3[0].dims)
    o5_ndim = tuple(oTensor_5[0].dims)
    o7_ndim = tuple(oTensor_7[0].dims)

    batchSize = i1_ndim[0]
    
    out1 = numpy.zeros(o1_ndim, dtype='float32')
    out3 = numpy.zeros(o3_ndim, dtype='float32')
    out5 = numpy.zeros(o5_ndim, dtype='float32')
    out7 = numpy.zeros(o7_ndim, dtype='float32')

    n_of_images = len(img)
    count = 0
    write_index = 0
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images - count
        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [numpy.empty(i1_ndim, dtype=numpy.float32, order="C")]
        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(i1_ndim[1:])
        '''run with batch '''
        execute_async(dpu_1, {
            get_DPU_iName(dpu_1): inputData[0],
            get_DPU_oName(dpu_1): out1})
        
        inp2 = out1.copy()
        out2 = ReLU(inp2)
        
        execute_async(dpu_3, {
            get_DPU_iName(dpu_3): out2,
            get_DPU_oName(dpu_3): out3})
        
        inp4 = out3.copy()
        out4 = ReLU(inp4)

        execute_async(dpu_5, {
            get_DPU_iName(dpu_5): out4,
            get_DPU_oName(dpu_5): out5})
        
        inp6 = out5.copy()
        out6 = ReLU(inp6)

        execute_async(dpu_7, {
            get_DPU_iName(dpu_7): out6,
            get_DPU_oName(dpu_7): out7})

        inp8 = out7.copy()
        fout = LogSoftmax(inp6)

        '''store output vectors '''
        for j in range(runSize):
            out_q[write_index] = Argmax(fout[j])
            write_index += 1
        count = count + runSize



''' global list that all threads can write results to '''
global out_q
out_q = [None] * N_IMG
runTotal = N_IMG

g = xir.Graph.deserialize('./dwc_ob/dwc_ob.xmodel')
subgraph = g.get_root_subgraph().toposort_child_subgraph()
dpu_sg0 = subgraph[0]   # input

dpu_sg1 = subgraph[1]
dpu_sg3 = subgraph[3]
dpu_sg5 = subgraph[5]
dpu_sg7 = subgraph[7]
dpu_1 = vart.Runner.create_runner(dpu_sg1, "run")
dpu_3 = vart.Runner.create_runner(dpu_sg3, "run")
dpu_5 = vart.Runner.create_runner(dpu_sg5, "run")
dpu_7 = vart.Runner.create_runner(dpu_sg7, "run")
img = numpy.random.rand(N_IMG, 64, 64, 3)   # dummy input for test


time1 = time.time()
runDPU(dpu_1, dpu_3, dpu_5, dpu_7, img)
time2 = time.time()
timetotal = time2 - time1
fps = float(runTotal / timetotal)
print(" ")
print("FPS=%.2f, total frames = %.0f , time=%.4f seconds" %(fps, runTotal, timetotal))
print(" ")