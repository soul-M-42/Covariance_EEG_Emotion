datasets=('S1' 'S7' 'F' 'S1S7' 'FD' 'S1F' 'S1S4S7D' 'S1S7FD')

for dataset in "${datasets[@]}"
do
    echo "dataset=$dataset"
    python ext_fea.py log.run_name="FIXCDA_$dataset"
done