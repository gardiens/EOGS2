
cd src/gaussiansplatting
for scene in "IARPA_001" "IARPA_002" "IARPA_003" "JAX_004" "JAX_068" "JAX_214" "JAX_260"
do
    # deal with the Panchromatic with the 3PAN mode and the pansharpening with onlyMSI mode
    for pair in "pan 3PAN" "pansharpen onlyMSI"; do
        set -- $pair
        dataset=$1
        mode=$2

        
        # run base eogs
        python full_eval_pan.py experiments=base_eogs.yaml mode=$mode rpc_type=rpc_ba scene=$scene dataset=$dataset expname=baseeogs_rpcba_${scene}_${dataset}_${mode}
        # run EOGS++ with b.a. cameras
            python full_eval_pan.py experiments=eogsplusrpcba.yaml mode=$mode rpc_type=rpc_ba scene=$scene dataset=$dataset expname=eogsplus_rpcba_${scene}_${dataset}_${mode}
    
        # run EOGS++ with raw rpc cameras
        python full_eval_pan.py experiments=eogsplusrpcba.yaml mode=$mode rpc_type=rpc_raw scene=$scene dataset=$dataset expname=eogsplus_rawrpc_${scene}_${dataset}_${mode}
        
        # run EOGS++ with learn wv method on raw rpc  
        python full_eval_pan.py experiments=learnwv.yaml mode=$mode rpc_type=rpc_raw scene=$scene dataset=$dataset
        expname=learnwv_rawrpc_${scene}_${dataset}_${mode}
        done
done
# you can visualize the outputs in the outputs/ folder
