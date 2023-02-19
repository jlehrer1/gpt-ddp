export CONTEXTLENGTH=700
export BATCHSIZE=32
envsubst < kube.yaml | kubectl create -f -