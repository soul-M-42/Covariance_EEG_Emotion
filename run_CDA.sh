CDAs=(0.1 0.2)

for cda in "${CDAs[@]}"
do
    echo "CDA=$cda"
    python train_multi.py train.loss.CDA_factor=$cda log.run_name="CDA_ITE_$cda"\
            train.n_gpu_use=4\
            train.n_pairs=1024\
            train.max_epochs=20
done