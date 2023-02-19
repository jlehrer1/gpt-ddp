export CONTEXTLENGTH=512
export BATCHSIZE=16
envsubst < kube.yaml | kubectl create -f -