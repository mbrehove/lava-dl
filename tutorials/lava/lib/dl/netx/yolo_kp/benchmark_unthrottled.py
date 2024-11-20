#!/usr/bin/env python

import os
import yaml
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.proc import embedded_io as eio
from lava.proc import io
from lava.proc.cyclic_buffer.process import CyclicBuffer

from lava.lib.dl import netx
from lava.lib.dl import slayer
from lava.lib.dl.slayer import obd

from utils import DataGenerator, YOLOPredictor, nms, YOLOMonitor, DeltaEncoder

# from IPython.display import display, clear_output


loihi2_is_available = True
random_seq = False
randomize_seq = random_seq

if loihi2_is_available:

    os.environ["LOIHI_GEN"] = "N3C1"
    os.environ["NXSDKHOST"] = (
        "ncl-gdc-vpx-01.zpn.intel.com"  # this needs to be changed for user specific system
    )
    os.environ["NXOPTIONS"] = (
        "--pio-cfg-chip=0x4191"  # this is board specific, most relaxed config is 0x41FF
    )

    from lava.magma.compiler.subcompilers.nc.ncproc_compiler import (
        CompilerOptions,
        # CompilerLogs,
    )

    CompilerOptions.verbose = True
    CompilerOptions.show_resource_count = True
    CompilerOptions.log_resource_count = True
    # CompilerOptions.large_mpds_max_size = (1 << 14) - 10000  # 2000 dwords for spike buffer
    compression = io.encoder.Compression.DELTA_SPARSE_8
else:
    print(
        "Loihi2 compiler is not available in this system. "
        "This tutorial will execute on CPU backend."
    )
    compression = io.encoder.Compression.DENSE


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
    # num_layers=2,
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
    randomize_seq=randomize_seq,
    seq_len=inference_args.num_steps,
)
start_idx = np.random.randint(len(test_set)) if random_seq else 0

data_gen = DataGenerator(
    dataset=test_set,
    start_idx=start_idx,
    mean=pre_args.input_mean,
    std=pre_args.input_std,
)

quantize = netx.modules.Quantize(
    exp=6
)  # convert to fixed point representation with 6 bit of fraction
sender = io.injector.Injector(shape=net.inp.shape, buffer_size=128)

encoder = DeltaEncoder(
    vth=net.net_config["layer"][0]["neuron"]["vThMant"], spike_exp=0, num_bits=8
)

frames = []
annotations = []
raw_frames = []
for t in range(99):
    frame, annotation, raw_frame = data_gen()
    frame = quantize(frame)
    frame = encoder(frame)
    # if t % 32 != 0:
    #     frame *= 0
    frames.append(frame)
    annotations.append(annotation)
    raw_frames.append(raw_frame)
    # clear_output(wait=True)
    # display(frame)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(raw_frames[0].permute([1, 2, 0]))
ax[1].imshow(frames[0].transpose([1, 0, 2]))
ax[2].imshow(frames[1].transpose([1, 0, 2]))

plt.savefig("profile_figs/first_frame.png")


input_source = CyclicBuffer(
    first_frame=frames[0],
    replay_frames=np.array(frames[1:]).transpose([1, 2, 3, 0]),
    spike_exp=net.spike_exp,
)


num_steps = inference_args.num_steps

from lava.utils import loihi2_profiler

power_logger = loihi2_profiler.Loihi2Power(num_steps=num_steps)
runtime_logger = loihi2_profiler.Loihi2ExecutionTime()
memory_logger = loihi2_profiler.Loihi2Memory()
activity_logger = loihi2_profiler.Loihi2Activity()
spike_logger = loihi2_profiler.Loihi2Spike()

callback_fxs = [
    power_logger,
    runtime_logger,
    memory_logger,
    activity_logger,
    spike_logger,
]


from nxcore.arch.n3b.n3board import N3Board

pre_num_steps = 100000
static_power_logger = loihi2_profiler.Loihi2Power(num_steps=pre_num_steps)

if "BOARD" in os.environ.keys() or "NXSDKHOST" in os.environ.keys():
    board = N3Board(1, 0)
    static_power_logger.pre_run_callback(board, {})
    board.run(pre_num_steps)
    static_power_logger.post_run_callback(board, {})
    board.disconnect()
else:
    raise RuntimeError(
        "Consistent board is not being targetted. The static measurement may happen on different board. Consider setting BOARD or NXSDKHOST environment."
    )


run_condition = RunSteps(num_steps=num_steps)

if inference_args.loihi:
    exception_proc_model_map = {
        io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelSparse
    }
    run_config = Loihi2HwCfg(
        exception_proc_model_map=exception_proc_model_map,
        callback_fxs=callback_fxs,
    )

net._log_config.level = logging.INFO
net.run(condition=run_condition, run_cfg=run_config)
net.wait()
net.stop()


# runtime measurements
inference_rate = 1e6 / runtime_logger.avg_time_per_step
total_inference_time = num_steps * runtime_logger.avg_time_per_step * 1e-6
time_per_step = runtime_logger.avg_time_per_step
latency = runtime_logger.avg_time_per_step * len(net)

print(f"Time per timestep : {time_per_step:10.2f} us")
print(f"Latency           : {latency:10.2f} us")
print(f"Throughput        : {inference_rate:10.2f} fps")


# power measurements
vdd_p = power_logger.vdd_power  # neurocore power
vddm_p = power_logger.vddm_power  # memory power
vddio_p = power_logger.vddio_power  # IO power
total_power = power_logger.total_power

plt.figure(figsize=(15, 3))
plt.plot(static_power_logger.total_power, label="Static Power Profile")
plt.plot(
    static_power_logger.static_total_power
    * np.ones_like(static_power_logger.total_power),
    label="Pre execution baseline",
)
plt.plot(
    power_logger.static_total_power
    * np.ones_like(static_power_logger.total_power),
    label="Post execution baseline",
)
plt.ylim(0, 8)
plt.legend()
plt.savefig("profile_figs/power_results.png")

num_chips = 16
if "PARTITION" in os.environ.keys():
    if "kp" in os.environ["PARTITION"]:
        num_chips = 8

active_chips = (
    max(activity_logger.core_idx + 120 * activity_logger.chip_idx) + 119
) // 120
print(f"active chips : {active_chips}")
# per chip static power
static_total_power_per_chip = static_power_logger.static_total_power / num_chips
static_vdd_p_per_chip = static_power_logger.static_vdd_power / num_chips
static_vddm_p_per_chip = static_power_logger.static_vddm_power / num_chips
static_vddio_p_per_chip = static_power_logger.static_vddio_power / num_chips

# compensate for static power of unused chips
total_power -= (num_chips - active_chips) * static_total_power_per_chip
vdd_p -= (num_chips - active_chips) * static_vdd_p_per_chip
vddm_p -= (num_chips - active_chips) * static_vddm_p_per_chip
vddio_p -= (num_chips - active_chips) * static_vddio_p_per_chip

# active chips static power
static_total_power = (
    power_logger.static_total_power
    - (num_chips - active_chips) * static_total_power_per_chip
)
static_vdd_p = (
    power_logger.static_vdd_power
    - (num_chips - active_chips) * static_vdd_p_per_chip
)
static_vddm_p = (
    power_logger.static_vddm_power
    - (num_chips - active_chips) * static_vddm_p_per_chip
)
static_vddio_p = (
    power_logger.static_vddio_power
    - (num_chips - active_chips) * static_vddio_p_per_chip
)


total_power_mean = np.mean(total_power)
vdd_p_mean = np.mean(vdd_p)
vddm_p_mean = np.mean(vddm_p)
vddio_p_mean = np.mean(vddio_p)
dynamic_power_mean = total_power_mean - static_total_power

print(f"Total Power   : {total_power_mean:.6f} W")
print(f"Dynamic Power : {dynamic_power_mean:.6f} W")
print(f"Static Power  : {static_total_power:.6f} W")
print(f"VDD Power     : {vdd_p_mean:.6f} W")
print(f"VDD-M Power   : {vddm_p_mean:.6f} W")
print(f"VDD-IO Power  : {vddio_p_mean:.6f} W")


total_energy = total_power_mean / inference_rate
dynamic_energy = (total_power_mean - static_total_power) / inference_rate
print(f"Total Energy per inference   : {total_energy * 1e3:.6f} mJ")
print(f"Dynamic Energy per inference : {dynamic_energy * 1e3:.6f} mJ")


import itertools

ucode_passes = 2  # 2 passes for sdn_relu

# num_cores_per_layer = CompilerLogs.partition_distr[:, 7]
macs_per_layer = [
    4816896,
    231211008,
    25690112,
    231211008,
    115605504,
    231211008,
    57802752,
    57802752,
    57802752,
    21676032,
]
ann_macs = sum(macs_per_layer)
# ann_macs = np.array(
#     list(
#         itertools.chain.from_iterable(
#             [[m / n] * n for (m, n) in zip(macs_per_layer, num_cores_per_layer)]
#         )
#     )
# )
synops_per_frame = np.sum(activity_logger.syn_ops / num_steps)
ann_ops_per_frame = np.sum(ann_macs)
synops_sparsity_factor = ann_ops_per_frame / synops_per_frame

num_valid = len(ann_macs)
activity_sparsity_factor = (
    activity_logger.dendrite_updates[4:].sum()
    / activity_logger.spike_axon_in[4:].sum()
    / ucode_passes
)
activity_sparsity = np.ones(activity_logger.dendrite_updates.shape)
activity_sparsity[:num_valid] = (
    activity_logger.dendrite_updates[:num_valid]
    / activity_logger.spike_axon_in[:num_valid]
    / ucode_passes
)
activity_sparsity[:4] = (
    1  # first 4 cores are output cores that only contain sigma neurons
)

tops = synops_per_frame * inference_rate / 1e12
tops_watt = tops / total_power_mean

print(
    f"ANN MACs per frame     = {ann_ops_per_frame:14.2f} = {ann_ops_per_frame / 1e6:7.2f} MOps"
)
print(
    f"SDNN SynOps per frame  = {synops_per_frame:14.2f} = {synops_per_frame / 1e6:7.2f} MOps"
)
print(f"SDNN Activity Sparsity = {activity_sparsity_factor:.2f}x")
print(f"SDNN Synapse Sparsity  = {synops_sparsity_factor:.2f}x")
print(f"SDNN TOPS              = {tops:.3f}")
print(f"SDNN TOPS/W            = {tops_watt:.3f}")
print(f"ANN equiv. TOPS        = {synops_sparsity_factor * tops:.3f}")
print(f"ANN equiv. TOPS/W      = {synops_sparsity_factor * tops_watt:.3f}")


time_per_syn_op = 0.66e-3  # 0.66ns # N3B3
time_per_neuron_op = 4e-3  # 4ns
time_per_barrier_sync = 10  # 10us
# core computation time is the largest of time taken for syn_ops, neuron_ops or barrier_sync
core_computation_time = np.maximum(
    time_per_syn_op * activity_logger.syn_ops / num_steps,
    time_per_neuron_op * activity_logger.dendrite_updates / num_steps,
    time_per_barrier_sync * np.ones_like(activity_logger.syn_ops, dtype=float),
)
time_per_timestep_corebound = max(core_computation_time)

print(
    f"Est. core processing time per timestep = {time_per_timestep_corebound}us"
)


# User bandwidth across the PIO link is 8b*800MHz.
# A remote long spike is 3*4B long (remote header, spike header, spike activation).
# Thus graded spike rate over a PIO link is 8b*800MHz / 3*4*8b = 67Mspike/s.
# Binary spikes should be 100Mspike/s.
bridge_router_bandwidth = 800e6 / (3 * 4)
remote_spike_routing_time = (
    spike_logger.num_remote_spikes
    / activity_sparsity
    / bridge_router_bandwidth
    * 1e6
)  # us
for i in range(0, len(spike_logger.num_remote_spikes), 120):
    remote_spike_routing_time[i : i + 120] = np.sum(
        remote_spike_routing_time[i : i + 120]
    )

print(
    f"Est. remote spike routing time per timestep = {np.max(remote_spike_routing_time)}us"
)

spike_logger.max_remote_fanout.max()

avg_memory_utilization = np.mean(memory_logger.total_mpds)
print(f"Avg. neurocore memory utilization: {avg_memory_utilization * 100:.2f}%")


fig = plt.figure(figsize=(15, 17))

layer_start = np.cumsum(num_cores_per_layer) - num_cores_per_layer
layers = [
    Rectangle((start, -1e10), num, 2e10)
    for start, num in zip(layer_start[::2], num_cores_per_layer[::2])
]
pc = PatchCollection(layers, facecolor="k", alpha=0.05)

gs = gridspec.GridSpec(18, 1)
gs.update(wspace=0.025, hspace=0.75)
ax_time = plt.subplot(gs[0:2, 0])
ax_power = plt.subplot(gs[2:6, 0])
ax_activity_syn = plt.subplot(gs[6:8, 0])
ax_activity = plt.subplot(gs[8:10, 0])
ax_sparsity = plt.subplot(gs[10:12, 0])
ax_spikes = plt.subplot(gs[12:14, 0])
ax_compute = plt.subplot(gs[14:16, 0])
ax_memory = plt.subplot(gs[16:18, 0])
ax_activity_syn.sharex(ax_activity)
ax_activity.sharex(ax_sparsity)
ax_sparsity.sharex(ax_compute)
ax_compute.sharex(ax_memory)

# Runtime Profile
ax_time.plot(
    runtime_logger.execution_time_per_step,
    label="Total execution time per timestep",
)
ax_time.plot(
    runtime_logger.management_time_per_step,
    label="Management time per timestep",
)
ax_time.plot(runtime_logger.host_time_per_step, label="Host time per timestep")
ax_time.set_xlabel("Time Step", labelpad=-10)
ax_time.set_ylabel("Time ($\mu$s)")
ax_time.legend()

# Power Profile
color = next(ax_power._get_lines.prop_cycler)["color"]
ax_power.plot(total_power, color=color, label="Total Power")
ax_power.plot(
    np.zeros_like(total_power) + static_total_power,
    linestyle="--",
    color=color,
    label="Total Static Power",
)
color = next(ax_power._get_lines.prop_cycler)["color"]
ax_power.plot(vdd_p, color=color, label="VDD Power")
color = next(ax_power._get_lines.prop_cycler)["color"]
ax_power.plot(vddm_p, color=color, label="VDD-M Power")
color = next(ax_power._get_lines.prop_cycler)["color"]
ax_power.plot(vddio_p, color=color, label="VDD-IO Power")
ax_power.set_ylabel("Power (W)")
ax_power.set_xlabel("Time (us)", labelpad=-10)
ax_power.legend()

# Activity Profile
logical_core_idx = activity_logger.core_idx + 120 * activity_logger.chip_idx
ax_activity_syn.plot(
    logical_core_idx, activity_logger.syn_ops / num_steps, label="Synaptic Ops"
)
ax_activity_syn.plot(
    logical_core_idx[: len(ann_macs)],
    ann_macs,
    linestyle="--",
    label="ANN MACs",
)
# ax_activity_syn.set_xlabel('Core Idx', labelpad=-10)
ax_activity_syn.set_ylabel("Count")
ax_activity_syn.legend()
ax_activity_syn.add_collection(
    PatchCollection(layers, facecolor="k", alpha=0.05)
)

ax_activity.plot(
    logical_core_idx,
    activity_logger.dendrite_updates / num_steps,
    label="Neuron Updates",
)
ax_activity.plot(
    logical_core_idx,
    activity_logger.axon_out / num_steps,
    label="Output Spikes",
)
ax_activity.plot(
    logical_core_idx,
    activity_logger.spikes_in / num_steps,
    label="Input Spikes",
)
# ax_activity.set_xlabel('Core Idx', labelpad=-10)
ax_activity.set_ylabel("Count")
ax_activity.legend()
ax_activity.add_collection(PatchCollection(layers, facecolor="k", alpha=0.05))

ax_sparsity.plot(
    logical_core_idx[:num_valid],
    ann_macs * num_steps / activity_logger.syn_ops[:num_valid],
    label="Synapse Sparsity (x)",
)
ax_sparsity.plot(
    logical_core_idx[:num_valid],
    activity_logger.dendrite_updates[:num_valid]
    / activity_logger.spike_axon_in[:num_valid]
    / ucode_passes,
    label="Activity Sparsity (x)",
)
# ax_sparsity.set_xlabel('Core Idx', labelpad=-10)
ax_sparsity.set_ylabel("Sparsity")
ax_sparsity.legend()
ax_sparsity.add_collection(PatchCollection(layers, facecolor="k", alpha=0.05))

logical_core_idx = spike_logger.core_idx + 120 * spike_logger.chip_idx
remote_spikes = spike_logger.num_remote_spikes
ax_spikes.plot(
    logical_core_idx,
    spike_logger.num_spikes / activity_sparsity,
    label="Total Spikes out of core",
)
ax_spikes.plot(
    logical_core_idx,
    spike_logger.num_remote_spikes / activity_sparsity,
    label="Remote Spikes",
)
ax_spikes.plot(
    logical_core_idx,
    spike_logger.num_local_spikes / activity_sparsity,
    label="Local Spikes",
)
ax_spikes.set_xlabel("Core Idx", labelpad=-10)
ax_spikes.set_ylabel("Count")
ax_spikes.legend()
ax_spikes.add_collection(PatchCollection(layers, facecolor="k", alpha=0.05))

ax_compute.plot(
    logical_core_idx,
    core_computation_time,
    label="Avg. core computation time per timestep (Estimated)",
)
# ax_compute.plot(logical_core_idx, time_per_step * np.ones_like(core_computation_time), linestyle='--', label='Avg. time per timestep')
ax_compute.plot(
    logical_core_idx,
    remote_spike_routing_time,
    label="Avg. remote spike routing time per timestep (Estimated)",
)
# ax_compute.set_xlabel('Core Idx', labelpad=-10)
ax_compute.set_ylabel("Time ($\mu$s)")
ax_compute.legend()
ax_compute.add_collection(PatchCollection(layers, facecolor="k", alpha=0.05))

# Memory Profile
logical_core_idx = memory_logger.core_idx + 120 * memory_logger.chip_idx
ax_memory.plot(
    logical_core_idx, memory_logger.total_mpds, label="Memory Utilization"
)
ax_memory.set_ylabel("Used / Total")
ax_memory.set_xlabel("Core Idx")
ax_memory.legend()
ax_memory.add_collection(PatchCollection(layers, facecolor="k", alpha=0.05))

for n in range(np.sum(num_cores_per_layer) // 120):
    ax_activity_syn.axvline(
        n * 120 + 120, linestyle="--", color="k", alpha=0.25
    )
    ax_activity.axvline(n * 120 + 120, linestyle="--", color="k", alpha=0.25)
    ax_sparsity.axvline(n * 120 + 120, linestyle="--", color="k", alpha=0.25)
    ax_spikes.axvline(n * 120 + 120, linestyle="--", color="k", alpha=0.25)
    ax_compute.axvline(n * 120 + 120, linestyle="--", color="k", alpha=0.25)
    ax_memory.axvline(n * 120 + 120, linestyle="--", color="k", alpha=0.25)

plt.savefig("profile_figs/activity.png")

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(
    logical_core_idx,
    spike_logger.max_remote_fanout,
    label="Total Spikes out of core",
)
ax.plot(
    logical_core_idx,
    spike_logger.min_remote_fanout,
    label="Total Spikes out of core",
)
ax.plot(
    logical_core_idx,
    spike_logger.num_remote_fanout,
    label="Total Spikes out of core",
)

plt.savefig("profile_figs/fanout.png")
# In[33]:


# from IPython.display import display, clear_output, Markdown
mAP50 = 0.194579
report = f"""
**Video Object Detection on BDD100K Benchmark**
| | YOLO-KP SDNN (Loihi 2, Unthrottled) |
|-|-:|
|mAP@50                           | {mAP50 * 100:.3f}%    |
|Activity Sparsity                | {activity_sparsity_factor:.3f}x     |
|SynOps Sparsity                  | {synops_sparsity_factor:.3f}x     |
|Latency ($\mu$s)                 | {latency:.3f} |
|Inference Throughput (fps)       | {inference_rate:.3f}     |
|Time per timestep ($\mu$s)       | {time_per_step:.3f}  |
|Time per timestep (Est.) ($\mu$s)| {max(time_per_timestep_corebound, np.max(remote_spike_routing_time)):.3f}   |
|Total Power (W)                  | {total_power_mean:.3f}      |
|Dynamic Power (W)                | {dynamic_power_mean:.3f}      |
|Total Energy per Frame (mJ)      | {total_energy * 1e3:.3f}     |
|Dynamic Energy per Frame (mJ)    | {dynamic_energy * 1e3:.3f}      |
|Equivalent TOPS                  | {synops_sparsity_factor * tops:.3f}      |
|Equivalent TOPS/W                | {synops_sparsity_factor * tops_watt:.3f}      |
|Total EDP per Frame ($\mu$J-s)   | {total_energy * latency:.3f}   |
|Dynamic EDP per Frame ($\mu$J-s) | {dynamic_energy * latency:.3f}    |
"""
# display(Markdown(report))
print(report)
