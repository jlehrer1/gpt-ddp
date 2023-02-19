export CONTEXTLENGTH=450
export BATCHSIZE=32
envsubst < kube.yaml | kubectl create -f -