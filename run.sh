export BATCHSIZE=4
export CONTEXTLENGTH=512
export NBLOCKS=12
export NHEADS=8
export NEMBD=128
envsubst < kube.yaml | kubectl create -f -