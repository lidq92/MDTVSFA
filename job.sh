#!/bin/bash

# Using K80 GPU

# ./job.sh -g 0 -d K > K-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 1 -d K -d C > KC-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d K -d L > KL-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d K -d N > KN-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d K -d C -d L > KCL-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 5 -d K -d C -d N > KCN-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 6 -d K -d L -d N > KLN-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 7 -d K -d C -d L -d N > KCLN-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 8 -d C > C-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 9 -d L > L-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 10 -d N > N-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 11 -d C -d L > CL-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 12 -d C -d N > CN-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 13 -d L -d N > LN-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 14 -d C -d L -d N > CLN-mixed-exp-0-10-1e-4-32-40.log 2>&1 &

# ./job.sh -g 15 -d K -d C -d L -d N -l naive > KCLN-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 14 -d C -l naive > C-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 13 -d L -l naive > L-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 12 -d N -l naive > N-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 11 -d C -d L -l naive > CL-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 10 -d C -d N -l naive > CN-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 9 -d L -d N -l naive > LN-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 8 -d C -d L -d N -l naive > CLN-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 7 -d K -l naive > K-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 6 -d K -d C -l naive > KC-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 5 -d K -d L -l naive > KL-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d K -d N -l naive > KN-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d K -d C -d L -l naive > KCL-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d K -d C -d N -l naive > KCN-naive-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 1 -d K -d L -d N -l naive > KLN-naive-exp-0-10-1e-4-32-40.log 2>&1 &

# ./job.sh -g 15 -d K -d C -d L -d N -l naive0 > KCLN-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 14 -d C -l naive0 > C-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 13 -d L -l naive0 > L-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 12 -d N -l naive0 > N-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 11 -d C -d L -l naive0 > CL-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 10 -d C -d N -l naive0 > CN-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 9 -d L -d N -l naive0 > LN-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 8 -d C -d L -d N -l naive0 > CLN-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 7 -d K -l naive0 > K-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 6 -d K -d C -l naive0 > KC-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 5 -d K -d L -l naive0 > KL-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d K -d N -l naive0 > KN-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d K -d C -d L -l naive0 > KCL-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d K -d C -d N -l naive0 > KCN-naive0-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 1 -d K -d L -d N -l naive0 > KLN-naive0-exp-0-10-1e-4-32-40.log 2>&1 &

# ./job.sh -g 0 -d K -d C -d L -d N -l plcc > KCLN-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 1 -d C -l plcc > C-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d L -l plcc > L-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d N -l plcc > N-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d C -d L -l plcc > CL-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 5 -d C -d N -l plcc > CN-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 6 -d L -d N -l plcc > LN-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 7 -d C -d L -d N -l plcc > CLN-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 8 -d K -l plcc > K-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 9 -d K -d C -l plcc > KC-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 10 -d K -d L -l plcc > KL-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 11 -d K -d N -l plcc > KN-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 12 -d K -d C -d L -l plcc > KCL-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 13 -d K -d C -d N -l plcc > KCN-plcc-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 14 -d K -d L -d N -l plcc > KLN-plcc-exp-0-10-1e-4-32-40.log 2>&1 &

# ./job.sh -g 15 -d K -d C -d L -d N -l rank > KCLN-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 14 -d C -l rank > C-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 13 -d L -l rank > L-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 12 -d N -l rank > N-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 11 -d C -d L -l rank > CL-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 10 -d C -d N -l rank > CN-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 9 -d L -d N -l rank > LN-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 8 -d C -d L -d N -l rank > CLN-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 7 -d K -l rank > K-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 6 -d K -d C -l rank > KC-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 5 -d K -d L -l rank > KL-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d K -d N -l rank > KN-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d K -d C -d L -l rank > KCL-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d K -d C -d N -l rank > KCN-rank-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 1 -d K -d L -d N -l rank > KLN-rank-exp-0-10-1e-4-32-40.log 2>&1 &

# ./job.sh -g 0 -d K -d C -d L -d N -l l1 > KCLN-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 1 -d C -l l1 > C-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d L -l l1 > L-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d N -l l1 > N-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d C -d L -l l1 > CL-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 5 -d C -d N -l l1 > CN-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 6 -d L -d N -l l1 > LN-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 7 -d C -d L -d N -l l1 > CLN-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 8 -d K -l l1 > K-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 9 -d K -d C -l l1 > KC-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 10 -d K -d L -l l1 > KL-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 11 -d K -d N -l l1 > KN-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 12 -d K -d C -d L -l l1 > KCL-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 13 -d K -d C -d N -l l1 > KCN-l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 14 -d K -d L -d N -l l1 > KLN-l1-exp-0-10-1e-4-32-40.log 2>&1 &

# ./job.sh -g 15 -d K -d C -d L -d N -l correlation > KCLN-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 14 -d C -l correlation > C-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 13 -d L -l correlation > L-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 12 -d N -l correlation > N-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 11 -d C -d L -l correlation > CL-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 10 -d C -d N -l correlation > CN-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 9 -d L -d N -l correlation > LN-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 8 -d C -d L -d N -l correlation > CLN-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 7 -d K -l correlation > K-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 6 -d K -d C -l correlation > KC-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 5 -d K -d L -l correlation > KL-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d K -d N -l correlation > KN-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d K -d C -d L -l correlation > KCL-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d K -d C -d N -l correlation > KCN-correlation-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 1 -d K -d L -d N -l correlation > KLN-correlation-exp-0-10-1e-4-32-40.log 2>&1 &

# ./job.sh -g 0 -d K -d C -d L -d N -l rank+l1 > KCLN-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 1 -d C -l rank+l1 > C-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d L -l rank+l1 > L-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d N -l rank+l1 > N-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d C -d L -l rank+l1 > CL-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 5 -d C -d N -l rank+l1 > CN-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 6 -d L -d N -l rank+l1 > LN-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 7 -d C -d L -d N -l rank+l1 > CLN-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 8 -d K -l rank+l1 > K-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 9 -d K -d C -l rank+l1 > KC-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 10 -d K -d L -l rank+l1 > KL-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 11 -d K -d N -l rank+l1 > KN-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 12 -d K -d C -d L -l rank+l1 > KCL-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 13 -d K -d C -d N -l rank+l1 > KCN-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 14 -d K -d L -d N -l rank+l1 > KLN-rank+l1-exp-0-10-1e-4-32-40.log 2>&1 &

# ./job.sh -g 15 -d K -d C -d L -d N -l plcc+l1 > KCLN-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 14 -d C -l plcc+l1 > C-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 13 -d L -l plcc+l1 > L-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 12 -d N -l plcc+l1 > N-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 11 -d C -d L -l plcc+l1 > CL-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 10 -d C -d N -l plcc+l1 > CN-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 9 -d L -d N -l plcc+l1 > LN-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 8 -d C -d L -d N -l plcc+l1 > CLN-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 7 -d K -l plcc+l1 > K-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 6 -d K -d C -l plcc+l1 > KC-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 5 -d K -d L -l plcc+l1 > KL-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d K -d N -l plcc+l1 > KN-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d K -d C -d L -l plcc+l1 > KCL-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d K -d C -d N -l plcc+l1 > KCN-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 1 -d K -d L -d N -l plcc+l1 > KLN-plcc+l1-exp-0-10-1e-4-32-40.log 2>&1 &

# ./job.sh -g 1 -d K -d C -d L -d N -p 1 > KCLN-mixed-train_proportion=1-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 2 -d K -d C -d L -d N -p 2 > KCLN-mixed-train_proportion=2-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 3 -d K -d C -d L -d N -p 3 > KCLN-mixed-train_proportion=3-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 4 -d K -d C -d L -d N -p 4 > KCLN-mixed-train_proportion=4-exp-0-10-1e-4-32-40.log 2>&1 &
# ./job.sh -g 0 -d K -d C -d L -d N -p 5 > KCLN-mixed-train_proportion=5-exp-0-10-1e-4-32-40.log 2>&1 &

loss=mixed
start_id=0
end_id=10
train_proportion=6
while getopts "g:d:l:s:e:p:" opt; do
    case $opt in
    	g) gpu_id=("$OPTARG");; # gpu_id
        d) datasets+=("$OPTARG");; # trained datasets
    	l) loss=("$OPTARG");; # loss
        s) start_id=("$OPTARG");;
        e) end_id=("$OPTARG");;
        p) train_proportion=("$OPTARG");;
    esac
done
shift $((OPTIND -1))
# if [ ! $loss ]; then
#     loss=mixed
# fi
# echo $loss
source activate reproducibleresearch
for ((i=$start_id; i<$end_id; i++)); do
	CUDA_VISIBLE_DEVICES=$gpu_id python main.py --exp_id=$i --train_proportion $train_proportion --loss=$loss --lr=1e-4 --batch_size=32 --epochs=40 --trained_datasets ${datasets[@]}
done
source deactivate 