#!/bin/bash

export SLURM=1
export LOIHI_GEN=N3C1
export PARTITION=vpx
export BOARD=ncl-ext-vpx-01
export NUM_CHIPS_CONSTRAINT=-Q
export NXOPTIONS="--pio-cfg-chip=0x4192"

cap_net_raw --ambient-caps +net_raw python3.10 benchmark_unthrottled.py