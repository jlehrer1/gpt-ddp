export BATCHSIZE=16
export CONTEXTLENGTH=256
export NBLOCKS=12
export NHEADS=8
export NEMBD=128
export LR=5e-3
envsubst < kube.yaml | kubectl create -f -