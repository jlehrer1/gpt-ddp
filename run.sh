export BATCHSIZE=32
export CONTEXTLENGTH=256
export NLAYERS=24
export NHEADS=4
export NEMBD=512
export LR=3e-4
export ACCUMBATCHES=0
export NUMWORKERS=32
export WARMUP=0

envsubst < kube.yaml | kubectl create -f -

