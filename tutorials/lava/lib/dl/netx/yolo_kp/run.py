#!/usr/bin/env python
# coding: utf-8

# # YOLO-KP SDNN Example
#
# This tutorial demonstrates the inference of YOLO-KP SDNN (training example scripts [here](https://github.com/lava-nc/lava-dl/tree/main/tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn)) on both CPU and Loihi 2 neurocore.
#
# ![image](https://github.com/lava-nc/lava/assets/29907126/61057e64-71b3-4ab8-a7ea-39d0cdbac70d)
#
# YOLO-KP is a fully convolutional single-headed variant of TinyYOLOv3 object detection architecture specifically designed for 8 chip Loihi 2 form factor called Kapoho Point (KP). The inference example uses the following lava components
#
# 1. __Network on Loihi 2:__ YOLO-KP network generated from its NetX description. It is a hierarchical network consisting of all the layers of YOLO-KP. This is the portion that runs on Loihi.
# 2. __Data sparsification on SuperHost:__ Delta encoder process that performs frame difference to sparsify the input being communicated to the YOLO-KP network. This process runs on Python.
# 3. __Data communication in and out of lava processes:__ `Injector` process to send raw input to the lava network and `Extractor` process to receive raw output of YOLO-KP. These processes run on Python.
# 4. __Data relay in and out of Loihi chip:__ Input and output adapter process which relay the communication into the chip and out of the chip. Since YOLO-KP is fully convolutional, the adapters translate to/from python spike and Loihi convolution spike.
#
# > ℹ️ This example currently does not make use of high speed IO capabilities of Loihi and hence the execution is slow. Once the software support is enabled in Lava, these adapters will not be required and shall be removed.

# In[1]:


import os
import yaml
import logging

import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.proc import embedded_io as eio
from lava.proc import io

from lava.lib.dl import netx
from lava.lib.dl import slayer
from lava.lib.dl.slayer import obd

from utils import DataGenerator, YOLOPredictor, nms, YOLOMonitor
from IPython.display import display, clear_output


# # Import modules for Loihi2 execution
#
# Check if Loihi2 compiler is available and import related modules.

# In[2]:


from lava.utils.system import Loihi2

Loihi2.preferred_partition = "loihi"
loihi2_is_available = Loihi2.is_loihi2_available

if loihi2_is_available:
    print(f"Running on {Loihi2.partition}")
    from lava.magma.compiler.subcompilers.nc.ncproc_compiler import (
        CompilerOptions,
    )

    CompilerOptions.verbose = True
    compression = io.encoder.Compression.DELTA_SPARSE_8
else:
    print(
        "Loihi2 compiler is not available in this system. "
        "This tutorial will execute on CPU backend."
    )
    compression = io.encoder.Compression.DENSE


# ## Set execution parameters
#
# The network execution parameters can be divided into three categories:
#
# 1. __Model parameters:__ these are parameters of the YOLO model used for the training and shall be reused to replicate the same behavior during inference.
# 2. __Inference parametrs:__ these are parameters just for the inference.
# 3. __Data processing parameters:__ these are parameters need to perform _pre_ and _post_ processing before the input and on the output of the network respectively.

# In[3]:


# Model arguments
# trained_folder = os.path.abspath('../../slayer/tiny_yolo_sdnn/Trained_yolo_kp')
from pathlib import Path

trained_folder = str(
    Path(__file__).parent.parent.parent
    / "slayer/tiny_yolo_sdnn/Trained_yolo_kp"
)
with open(trained_folder + "/args.txt", "rt") as f:
    model_args = slayer.utils.dotdict(yaml.safe_load(f))

# Additional inference arguments
inference_args = slayer.utils.dotdict(
    loihi=loihi2_is_available,
    spike_exp=4,  # This sets the decimal/fraction precision of spike message to 4 bits
    num_steps=100,
)  # Number of frames to perform inference on

# Pre and post processing parameters
pre_args = slayer.utils.dotdict(
    input_mean=np.array([0.485, 0.456, 0.406]),  # Input normalization mean
    input_std=np.array([0.229, 0.224, 0.225]),
)  #                     & std
post_args = slayer.utils.dotdict(
    anchors=np.array(
        [
            (0.28, 0.22),  # YOLO head's anchor preset scales
            (0.38, 0.48),
            (0.90, 0.78),
        ]
    )
)


# ## Load YOLO-KP network
#
# Loading the network is a simple NetX call on the trained model computational graph. It will generate an hierarchical lava process representing the entire YOLO-KP network.

# In[4]:


net = netx.hdf5.Network(
    trained_folder + "/network.net",
    skip_layers=1,  # First layer does delta encoding. We will only send it's sparsified output
    input_message_bits=16,  # This means the network takes 16bit graded spike input
    spike_exp=inference_args.spike_exp,
)
print(f"The model was trained for {model_args.dataset} dataset")
print(f"\nNetwork Architecture ({model_args.model}):")
print("=" * (24 + len(model_args.model)))
print(net)


# ## Dataset and input source
#
# The dataset is the same module that is used for training. It is wrapped around by a data generator module that will generate an individual frame and its annotation at every time-step. The data generator also takes care of data normalization using the mean and variance supplied.

# In[5]:


test_set = obd.dataset.BDD(
    root=model_args.path,
    dataset="track",
    train=False,
    randomize_seq=False,
    seq_len=inference_args.num_steps,
)
test_set.datasets[0].ids = sorted(test_set.datasets[0].ids)  # for determinism
data_gen = DataGenerator(
    dataset=test_set, mean=pre_args.input_mean, std=pre_args.input_std
)


# ## Input preprocessing and encoding
#
# The input preprocessing involves quantization of the numeric data making it ready to be processed on the chip. A fractional representation of 6 bits was used in the weight of the network during training (`weight_exp`) which is also accounted for during quantization.
#
# The quantized input frames are then processed the the lava processes `sender`, `encoder` (and `inp_adapter` for Loihi execution) which will be connected in a sequential manner below.

# In[6]:


quantize = netx.modules.Quantize(
    exp=6
)  # convert to fixed point representation with 6 bit of fraction
sender = io.injector.Injector(shape=net.inp.shape, buffer_size=128)
encoder = io.encoder.DeltaEncoder(
    shape=net.inp.shape,
    vth=net.net_config["layer"][0]["neuron"]["vThMant"],
    spike_exp=0 if inference_args.loihi else net.spike_exp,
    num_bits=8,
    compression=compression,
)
if inference_args.loihi:
    # This is needed for the time being until high speed IO is enabled
    inp_adapter = eio.spike.PyToN3ConvAdapter(
        shape=encoder.s_out.shape,
        num_message_bits=16,
        spike_exp=net.spike_exp,
        compression=compression,
    )


# ## Output decoding and post processing
#
# The output of YOLO-KP goes through (`state_adapter` for Loihi execution), `receiver` and `dequantizer` lava processes which will be connected sequentially. The raw outputs needs to be processed using `yolo_predictor` which transforms the input to the actual bounding box predictions of the network.

# In[7]:


if inference_args.loihi:
    # This is needed for the time being until high speed IO is enabled
    state_adapter = eio.state.ReadConv(shape=net.out.shape)
receiver = io.extractor.Extractor(shape=net.out.shape, buffer_size=128)
dequantize = netx.modules.Dequantize(exp=net.spike_exp + 12, num_raw_bits=24)
yolo_predictor = YOLOPredictor(
    anchors=post_args.anchors, clamp_max=model_args.clamp_max
)


# ## Output visualization
#
# `YOLOMonitor` is a flexible output visualization and evaluation module. It continuously evaluates the mAP score of the output predictions. It can also be passed a callable function that can be used to display. In this case it is a basic iPython display routine.

# In[8]:


def output_visualizer(annotated_frame, map_score, frame_idx):
    clear_output(wait=True)
    display(annotated_frame)
    print(f"Processed frame {frame_idx}")
    print(f"Object detection mAP@0.5 = {map_score:.2f}")


yolo_monitor = YOLOMonitor(
    viz_fx=output_visualizer, class_list=test_set.classes
)


# ## Data buffers / delays
#
# There is a latency in the prediction equal to the number of layers the network has and the encoding step. Two FIFO buffers are used to synchronize the input frame and target annotation with the predicted output.

# In[9]:


frame_buffer = netx.modules.FIFO(depth=len(net) + 1)
annotation_buffer = netx.modules.FIFO(depth=len(net) + 1)


# # Connect Lava processes

# In[10]:


if inference_args.loihi:
    sender.out_port.connect(encoder.a_in)
    encoder.s_out.connect(inp_adapter.inp)
    inp_adapter.out.connect(net.inp)
    state_adapter.connect_var(net.out_layer.neuron.sigma)
    state_adapter.out.connect(receiver.in_port)
else:
    sender.out_port.connect(encoder.a_in)
    encoder.s_out.connect(net.inp)
    net.out.connect(receiver.in_port)


# # Setup execution
#
# The network is run in _non-blocking mode_. Note the `blocking=False` argument below. In non-blocking mode we can start running the lava process and do other computations in parallel. Here we will preprocess the data, send it to lava network using `sender` (`lava.proc.io.injector.Injector` instance), receive data from lava using `receiver` (`lava.proc.io.extractor.Extractor` instance), and perform additional processing, while the Lava network is running in parallel.

# In[11]:


num_steps = inference_args.num_steps
run_condition = RunSteps(num_steps=num_steps, blocking=False)

if inference_args.loihi:
    exception_proc_model_map = {
        io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelSparse
    }
    run_config = Loihi2HwCfg(exception_proc_model_map=exception_proc_model_map)
else:
    exception_proc_model_map = {
        io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelDense
    }
    run_config = Loihi2SimCfg(
        select_tag="fixed_pt", exception_proc_model_map=exception_proc_model_map
    )


# # Run YOLO-KP inference
#
# The following will compile and run the Lava network.
#
# > ℹ️ The network is large. It will take a while for the compilation to finish.

# In[12]:


sender._log_config.level = logging.WARN
sender.run(condition=run_condition, run_cfg=run_config)


# In[ ]:


for t in range(num_steps):
    frame, annotation, raw_frame = data_gen()
    frame = quantize(frame)

    sender.send(frame)  # This sends the input frame to the Lava network
    out = receiver.receive()  # This receives the output from the Lava network

    out = dequantize(out)
    input_frame = frame_buffer(raw_frame)
    gt_ann = annotation_buffer(annotation)
    if input_frame is not None:  # valid output from FIFO buffer
        predictions = yolo_predictor(out)
        pred_bbox = nms(predictions)
        gt_bbox = (
            obd.bbox.utils.tensor_from_annotation(gt_ann).cpu().data.numpy()
        )
        yolo_monitor(input_frame, gt_bbox, pred_bbox)
    else:
        print(f"Frame {t} queued in pipeline.", end="\r")

sender.wait()
sender.stop()
