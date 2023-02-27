export BATCHSIZE=32
export CONTEXTLENGTH=256
export NLAYERS=24
export NHEADS=4
export NEMBD=128
export LR=5e-3
export ACCUMBATCHES=0
export NUMWORKERS=32
export WARMUP=0
export NGPU=8
envsubst < kube.yaml | kubectl create -f -

