export BATCHSIZE=16
export CONTEXTLENGTH=256
export NBLOCKS=12
export NHEADS=8
export NEMBD=128
envsubst < kube.yaml | kubectl create -f -