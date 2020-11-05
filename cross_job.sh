#!/bin/bash

# Using K80 GPU

# ./cross_job.sh -g 0 -d K -c C -c L -c N -l mixed > Ktrained-crossCLN-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 1 -c K -d C -c L -c N -l mixed > Ctrained-crossKLN-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 2 -c K -c C -d L -c N -l mixed > Ltrained-crossKCN-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 3 -c K -c C -c L -d N -l mixed > Ntrained-crossKCL-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 4 -d K -d C -c L -c N -l mixed > KCtrained-crossLN-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 5 -d K -c C -d L -c N -l mixed > KLtrained-crossCN-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 6 -d K -c C -c L -d N -l mixed > KNtrained-crossCL-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 7 -c K -d C -d L -c N -l mixed > CLtrained-crossKN-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 8 -c K -d C -c L -d N -l mixed > Ntrained-crossKL-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 9 -c K -c C -d L -d N -l mixed > Ltrained-crossKL-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 10 -d K -d C -d L -c N -l mixed > KCLtrained-crossN-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 11 -d K -d C -c L -d N -l mixed > KCNtrained-crossL-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 12 -c K -d C -d L -d N -l mixed > CLNtrained-crossK-mixed-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 13 -d K -c C -d L -d N -l mixed > KLNtrained-crossC-mixed-exp-0-10.log 2>&1 &

# ./cross_job.sh -g 0 -d K -c C -c L -c N -l naive > Ktrained-crossCLN-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 1 -c K -d C -c L -c N -l naive > Ctrained-crossKLN-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 2 -c K -c C -d L -c N -l naive > Ltrained-crossKCN-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 3 -c K -c C -c L -d N -l naive > Ntrained-crossKCL-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 4 -d K -d C -c L -c N -l naive > KCtrained-crossLN-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 5 -d K -c C -d L -c N -l naive > KLtrained-crossCN-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 6 -d K -c C -c L -d N -l naive > KNtrained-crossCL-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 7 -c K -d C -d L -c N -l naive > CLtrained-crossKN-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 8 -c K -d C -c L -d N -l naive > Ntrained-crossKL-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 9 -c K -c C -d L -d N -l naive > Ltrained-crossKL-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 10 -d K -d C -d L -c N -l naive > KCLtrained-crossN-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 11 -d K -d C -c L -d N -l naive > KCNtrained-crossL-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 12 -c K -d C -d L -d N -l naive > CLNtrained-crossK-naive-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 13 -d K -c C -d L -d N -l naive > KLNtrained-crossC-naive-exp-0-10.log 2>&1 &

# ./cross_job.sh -g 0 -d K -c C -c L -c N -l naive0 > Ktrained-crossCLN-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 1 -c K -d C -c L -c N -l naive0 > Ctrained-crossKLN-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 2 -c K -c C -d L -c N -l naive0 > Ltrained-crossKCN-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 3 -c K -c C -c L -d N -l naive0 > Ntrained-crossKCL-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 4 -d K -d C -c L -c N -l naive0 > KCtrained-crossLN-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 5 -d K -c C -d L -c N -l naive0 > KLtrained-crossCN-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 6 -d K -c C -c L -d N -l naive0 > KNtrained-crossCL-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 7 -c K -d C -d L -c N -l naive0 > CLtrained-crossKN-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 8 -c K -d C -c L -d N -l naive0 > Ntrained-crossKL-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 9 -c K -c C -d L -d N -l naive0 > Ltrained-crossKL-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 10 -d K -d C -d L -c N -l naive0 > KCLtrained-crossN-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 11 -d K -d C -c L -d N -l naive0 > KCNtrained-crossL-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 12 -c K -d C -d L -d N -l naive0 > CLNtrained-crossK-naive0-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 13 -d K -c C -d L -d N -l naive0 > KLNtrained-crossC-naive0-exp-0-10.log 2>&1 &

# ./cross_job.sh -g 0 -d K -c C -c L -c N -l l1 > Ktrained-crossCLN-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 1 -c K -d C -c L -c N -l l1 > Ctrained-crossKLN-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 2 -c K -c C -d L -c N -l l1 > Ltrained-crossKCN-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 3 -c K -c C -c L -d N -l l1 > Ntrained-crossKCL-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 4 -d K -d C -c L -c N -l l1 > KCtrained-crossLN-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 5 -d K -c C -d L -c N -l l1 > KLtrained-crossCN-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 6 -d K -c C -c L -d N -l l1 > KNtrained-crossCL-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 7 -c K -d C -d L -c N -l l1 > CLtrained-crossKN-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 8 -c K -d C -c L -d N -l l1 > Ntrained-crossKL-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 9 -c K -c C -d L -d N -l l1 > Ltrained-crossKL-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 10 -d K -d C -d L -c N -l l1 > KCLtrained-crossN-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 11 -d K -d C -c L -d N -l l1 > KCNtrained-crossL-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 12 -c K -d C -d L -d N -l l1 > CLNtrained-crossK-l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 13 -d K -c C -d L -d N -l l1 > KLNtrained-crossC-l1-exp-0-10.log 2>&1 &

# ./cross_job.sh -g 0 -d K -c C -c L -c N -l rank > Ktrained-crossCLN-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 1 -c K -d C -c L -c N -l rank > Ctrained-crossKLN-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 2 -c K -c C -d L -c N -l rank > Ltrained-crossKCN-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 3 -c K -c C -c L -d N -l rank > Ntrained-crossKCL-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 4 -d K -d C -c L -c N -l rank > KCtrained-crossLN-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 5 -d K -c C -d L -c N -l rank > KLtrained-crossCN-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 6 -d K -c C -c L -d N -l rank > KNtrained-crossCL-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 7 -c K -d C -d L -c N -l rank > CLtrained-crossKN-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 8 -c K -d C -c L -d N -l rank > Ntrained-crossKL-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 9 -c K -c C -d L -d N -l rank > Ltrained-crossKL-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 10 -d K -d C -d L -c N -l rank > KCLtrained-crossN-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 11 -d K -d C -c L -d N -l rank > KCNtrained-crossL-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 12 -c K -d C -d L -d N -l rank > CLNtrained-crossK-rank-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 13 -d K -c C -d L -d N -l rank > KLNtrained-crossC-rank-exp-0-10.log 2>&1 &

# ./cross_job.sh -g 0 -d K -c C -c L -c N -l plcc > Ktrained-crossCLN-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 1 -c K -d C -c L -c N -l plcc > Ctrained-crossKLN-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 2 -c K -c C -d L -c N -l plcc > Ltrained-crossKCN-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 3 -c K -c C -c L -d N -l plcc > Ntrained-crossKCL-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 4 -d K -d C -c L -c N -l plcc > KCtrained-crossLN-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 5 -d K -c C -d L -c N -l plcc > KLtrained-crossCN-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 6 -d K -c C -c L -d N -l plcc > KNtrained-crossCL-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 7 -c K -d C -d L -c N -l plcc > CLtrained-crossKN-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 8 -c K -d C -c L -d N -l plcc > Ntrained-crossKL-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 9 -c K -c C -d L -d N -l plcc > Ltrained-crossKL-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 10 -d K -d C -d L -c N -l plcc > KCLtrained-crossN-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 11 -d K -d C -c L -d N -l plcc > KCNtrained-crossL-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 12 -c K -d C -d L -d N -l plcc > CLNtrained-crossK-plcc-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 13 -d K -c C -d L -d N -l plcc > KLNtrained-crossC-plcc-exp-0-10.log 2>&1 &

# ./cross_job.sh -g 0 -d K -c C -c L -c N -l rank+l1 > Ktrained-crossCLN-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 1 -c K -d C -c L -c N -l rank+l1 > Ctrained-crossKLN-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 2 -c K -c C -d L -c N -l rank+l1 > Ltrained-crossKCN-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 3 -c K -c C -c L -d N -l rank+l1 > Ntrained-crossKCL-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 4 -d K -d C -c L -c N -l rank+l1 > KCtrained-crossLN-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 5 -d K -c C -d L -c N -l rank+l1 > KLtrained-crossCN-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 6 -d K -c C -c L -d N -l rank+l1 > KNtrained-crossCL-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 7 -c K -d C -d L -c N -l rank+l1 > CLtrained-crossKN-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 8 -c K -d C -c L -d N -l rank+l1 > Ntrained-crossKL-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 9 -c K -c C -d L -d N -l rank+l1 > Ltrained-crossKL-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 10 -d K -d C -d L -c N -l rank+l1 > KCLtrained-crossN-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 11 -d K -d C -c L -d N -l rank+l1 > KCNtrained-crossL-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 12 -c K -d C -d L -d N -l rank+l1 > CLNtrained-crossK-rank+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 13 -d K -c C -d L -d N -l rank+l1 > KLNtrained-crossC-rank+l1-exp-0-10.log 2>&1 &

# ./cross_job.sh -g 15 -d K -c C -c L -c N -l plcc+l1 > Ktrained-crossCLN-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 14 -c K -d C -c L -c N -l plcc+l1 > Ctrained-crossKLN-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 13 -c K -c C -d L -c N -l plcc+l1 > Ltrained-crossKCN-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 12 -c K -c C -c L -d N -l plcc+l1 > Ntrained-crossKCL-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 11 -d K -d C -c L -c N -l plcc+l1 > KCtrained-crossLN-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 10 -d K -c C -d L -c N -l plcc+l1 > KLtrained-crossCN-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 9 -d K -c C -c L -d N -l plcc+l1 > KNtrained-crossCL-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 8 -c K -d C -d L -c N -l plcc+l1 > CLtrained-crossKN-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 7 -c K -d C -c L -d N -l plcc+l1 > Ntrained-crossKL-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 6 -c K -c C -d L -d N -l plcc+l1 > Ltrained-crossKL-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 5 -d K -d C -d L -c N -l plcc+l1 > KCLtrained-crossN-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 4 -d K -d C -c L -d N -l plcc+l1 > KCNtrained-crossL-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 3 -c K -d C -d L -d N -l plcc+l1 > CLNtrained-crossK-plcc+l1-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 2 -d K -c C -d L -d N -l plcc+l1 > KLNtrained-crossC-plcc+l1-exp-0-10.log 2>&1 &

# ./cross_job.sh -g 0 -d K -c C -c L -c N -l correlation > Ktrained-crossCLN-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 1 -c K -d C -c L -c N -l correlation > Ctrained-crossKLN-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 2 -c K -c C -d L -c N -l correlation > Ltrained-crossKCN-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 3 -c K -c C -c L -d N -l correlation > Ntrained-crossKCL-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 4 -d K -d C -c L -c N -l correlation > KCtrained-crossLN-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 5 -d K -c C -d L -c N -l correlation > KLtrained-crossCN-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 6 -d K -c C -c L -d N -l correlation > KNtrained-crossCL-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 7 -c K -d C -d L -c N -l correlation > CLtrained-crossKN-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 8 -c K -d C -c L -d N -l correlation > Ntrained-crossKL-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 9 -c K -c C -d L -d N -l correlation > Ltrained-crossKL-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 10 -d K -d C -d L -c N -l correlation > KCLtrained-crossN-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 11 -d K -d C -c L -d N -l correlation > KCNtrained-crossL-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 12 -c K -d C -d L -d N -l correlation > CLNtrained-crossK-correlation-exp-0-10.log 2>&1 &
# ./cross_job.sh -g 13 -d K -c C -d L -d N -l correlation > KLNtrained-crossC-correlation-exp-0-10.log 2>&1 &

loss=mixed
start_id=0
end_id=10
while getopts "g:d:l:s:e:c:" opt; do
    case $opt in
    	g) gpu_id=("$OPTARG");; # gpu_id
        d) datasets+=("$OPTARG");; # trained datasets
    	l) loss=("$OPTARG");; # loss
        s) start_id=("$OPTARG");;
        e) end_id=("$OPTARG");;
        c) cross_datasets+=("$OPTARG");; # trained datasets
    esac
done
shift $((OPTIND -1))

source activate reproducibleresearch
for ((i=$start_id; i<$end_id; i++)); do
	CUDA_VISIBLE_DEVICES=$gpu_id python cross_dataset_evaluation.py --exp_id=$i --loss=$loss --trained_datasets ${datasets[@]}  --cross_datasets ${cross_datasets[@]}
done
source deactivate 