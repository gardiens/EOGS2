#!/bin/bash

for scene in JAX_260  JAX_214 JAX_004 JAX_068  IARPA_001 IARPA_002 IARPA_003
do
    for rpc_type in rpc_ba rpc_raw
    do
        for mode in pan pansharpen
        do
            python -m scripts.dataset_creation.to_affine scene_name=${scene} rpc_type=${rpc_type} mode=${mode}
        done
    done
done