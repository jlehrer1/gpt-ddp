export BATCHSIZE=16
export CONTEXTLENGTH=256
export NLAYERS=24
export NHEADS=4
export NEMBD=256
export LR=3e-4
export ACCUMBATCHES=8
export NUMWORKERS=32
export WARMUP=4000

envsubst < kube.yaml | kubectl create -f -

