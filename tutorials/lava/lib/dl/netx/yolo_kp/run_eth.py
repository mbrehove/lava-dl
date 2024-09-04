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
from lava.proc import io

from lava.lib.dl import netx
from lava.lib.dl import slayer
from lava.lib.dl.slayer import obd

from utils import DataGenerator, YOLOPredictor, nms, YOLOMonitor, DeltaEncoder

# from jupyter_utils import Display
# from IPython.display import Markdown

import logging

logging.getLogger().setLevel(logging.INFO)

logging.basicConfig(
    filename="yolo_kp_run.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


loihi2_is_available = True

if loihi2_is_available:
    # The following is a no-SLURM environment variable setting
    # os.environ["LOIHI_GEN"] = "N3C1"
    # os.environ["NXSDKHOST"] = (
    #     "ncl-gdc-vpx-01.zpn.intel.com"  # this needs to be changed for user specific system
    # )
    # os.environ["NXOPTIONS"] = (
    #     "--pio-cfg-chip=0x41FF"  # this is board specific, most relaxed config is 0x41FF
    # )

    # # The folliwng is a SLURM environment variable setting example
    # os.environ["SLURM"] = "1"
    # os.environ["LOIHI_GEN"] = "N3C1"
    # os.environ["BOARD"] = "ncl-og-01"                         # this needs to be changed for user specific system
    # os.environ["NXOPTIONS"] = "--pio-cfg-chip=0x41FF"         # this is board specific, most relaxed config is 0x41FF

    from lava.magma.compiler.subcompilers.nc.ncproc_compiler import (
        CompilerOptions,
    )

    CompilerOptions.verbose = True
else:
    print(
        "Loihi2 compiler is not available in this system. "
        "This tutorial will execute on CPU backend."
    )


# Model arguments
trained_folder = os.path.abspath("../../slayer/tiny_yolo_sdnn/Trained_yolo_kp")
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

quantize = netx.modules.Quantize(
    exp=6
)  # convert to fixed point representation with 6 bit of fraction
sender = io.injector.Injector(shape=net.inp.shape, buffer_size=128)
encoder = DeltaEncoder(
    vth=net.net_config["layer"][0]["neuron"]["vThMant"],
    spike_exp=net.spike_exp,
    num_bits=8,
)

receiver = io.extractor.Extractor(shape=net.out.shape, buffer_size=128)
dequantize = netx.modules.Dequantize(exp=net.spike_exp + 12, num_raw_bits=24)
yolo_predictor = YOLOPredictor(
    anchors=post_args.anchors, clamp_max=model_args.clamp_max
)

# display = Display(default_size=(448 * 2, 448))

# def output_visualizer(annotated_frame, map_score, frame_idx):
#     display.push(frame=annotated_frame,
#                  msg=f'Processed frame `{frame_idx}`\nObject detection mAP@0.5: `{map_score:.2f}`')

# yolo_monitor = YOLOMonitor(viz_fx=output_visualizer, class_list=test_set.classes)


frame_buffer = netx.modules.FIFO(depth=len(net))
annotation_buffer = netx.modules.FIFO(depth=len(net))

if inference_args.loihi:
    # The following are configurations specific to 10G ethernet connection
    from lava.magma.core.process.ports.connection_config import (
        ConnectionConfig,
        SpikeIOInterface,
        SpikeIOMode,
    )

    connection_config = ConnectionConfig()
    connection_config.interface = SpikeIOInterface.ETHERNET
    connection_config.spike_io_mode = SpikeIOMode.FREE_RUNNING
    connection_config.ethernet_chip_id = (2, 2, 1)  # same for VPX16 boards
    connection_config.ethernet_chip_idx = 12  # same for VPX16 boards
    # connection_config.ethernet_mac_address = "0x80615f11b9d6"  # MAC address needs to be the address of 10G NIC in super host
    connection_config.ethernet_mac_address = "0x90e2ba01214c"  # MAC address needs to be the address of 10G NIC in super host
    connection_config.ethernet_interface = "enp1s0"

    # # Eventual API will be the following, but it does not currently work.
    # sender.out_port.connect(net.inp)
    # net.out.connect(receiver.in_port)
    sender.out_port.connect(net.in_layer.synapse.s_in, connection_config)
    net.out_layer.neuron.s_out.connect(receiver.in_port, connection_config)
else:
    sender.out_port.connect(net.inp)
    net.out.connect(receiver.in_port)


num_steps = inference_args.num_steps
run_condition = RunSteps(num_steps=num_steps, blocking=False)

from lava.magma.core.model.py.model import PyLoihiModelToPyAsyncModel

# Async processes for fastest possible execution
exception_proc_model_map = {
    io.encoder.DeltaEncoder: PyLoihiModelToPyAsyncModel(
        io.encoder.PyDeltaEncoderModelDense
    ),
    io.injector.Injector: io.injector.PyLoihiInjectorModelAsync,
    io.extractor.Extractor: io.extractor.PyLoihiExtractorModelAsync,
}

if inference_args.loihi:
    run_config = Loihi2HwCfg(exception_proc_model_map=exception_proc_model_map)
else:
    run_config = Loihi2SimCfg(
        select_tag="fixed_pt", exception_proc_model_map=exception_proc_model_map
    )


print(
    "Running the inference on Loihi 2"
    if inference_args.loihi
    else "Running the inference on CPU"
)
sender._log_config.level = logging.INFO

# cache_folder = os.path.join(os.path.dirname(__file__), "cache_folder")
# os.makedirs(cache_folder, exist_ok=True)
# compile_config = {"cache": True, "cache_dir": cache_folder}
# sender.run(
#     condition=run_condition, run_cfg=run_config, compile_config=compile_config
# )
print("ready to compile.")
sender.run(
    condition=run_condition,
    run_cfg=run_config,
)
print("compiled.")
import threading
import queue


def sender_thread(sender, send_frame_queue, num_steps):
    # This is a dedicated routine to send data to the network as fast as possible
    for i in range(num_steps):
        frame = send_frame_queue.get()
        sender.send(frame)  # This sends the input frame to the Lava network
        print(f"Frame {i} sent to pipeline.")


def receiver_thread(receiver, recv_frame_queue, num_steps):
    # This is a dedicated routine to receive data from the network as fast as possible
    for t in range(num_steps):
        out = (
            receiver.receive()
        )  # This receives the output from the Lava network
        out = dequantize(out)

        annotation, raw_frame = recv_frame_queue.get()
        input_frame = frame_buffer(raw_frame)
        gt_ann = annotation_buffer(annotation)
        if input_frame is not None:  # valid output from FIFO buffer
            predictions = yolo_predictor(out)
            pred_bbox = nms(predictions)
            gt_bbox = (
                obd.bbox.utils.tensor_from_annotation(gt_ann).cpu().data.numpy()
            )
            # yolo_monitor(input_frame, gt_bbox, pred_bbox)
            print(f"Frame {t} processed.")
        else:
            print(f"Frame {t} queued in pipeline.")


s_q = queue.Queue()
r_q = queue.Queue()
s_th = threading.Thread(target=sender_thread, args=(sender, s_q, num_steps))
r_th = threading.Thread(target=receiver_thread, args=(receiver, r_q, num_steps))

s_th.start()
r_th.start()
# display.start()

for t in range(num_steps):
    frame, annotation, raw_frame = data_gen()
    frame = quantize(frame)
    frame = encoder(frame)

    s_q.put(frame)
    r_q.put((annotation, raw_frame))

s_th.join()
r_th.join()
print("Threads joined.")
sender.wait()
sender.stop()

# display.stop()
