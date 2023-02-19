CONTAINER = jmlehrer/gpt

.PHONY: build exec push run go train release 
exec:
	docker exec -it $(CONTAINER) /bin/bash

build:
	docker build -t $(CONTAINER) .

push:
	docker push $(CONTAINER)

run:
	docker run -it $(CONTAINER) /bin/bash

go:
	make build && make push

format:
	black --line-length=130 . && isort .