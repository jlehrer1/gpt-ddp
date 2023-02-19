export BATCHSIZE=32
export CONTEXTLENGTH=256
export NBLOCKS=12
export NHEADS=6
export NEMBD=128
envsubst < kube.yaml | kubectl create -f -