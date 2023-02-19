export CONTEXTLENGTH=64
export BATCHSIZE=32
envsubst < kube.yaml | kubectl create -f -