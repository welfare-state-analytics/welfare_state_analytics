.DEFAULT_GOAL=build

build:
	docker build -t sparv_v4:latest -f Dockerfile .
	# docker build -t rogermahler/sparv_v4:latest -f Dockerfile .
	# docker login docker.io
	# docker push rogermahler/sparv_v4:latest

bash:
	-docker run --entrypoint /bin/bash --rm -it -t sparv_v4:latest

bash2:
	-docker run --entrypoint /bin/bash \
		--user `id -u`:`id -g` \
		--workdir=/work \
		--mount "type=bind,src=`pwd`,dst=/work" \
		--volume="/etc/passwd:/etc/passwd:ro" \
		--volume="/etc/group:/etc/group:ro" \
		--volume="/etc/shadow:/etc/shadow:ro" \
		--rm -it sparv_v4:latest

prune:
	-docker container prune -f

killall:
	-docker rm `docker ps -aq`

rmi_untagged:
	-docker rmi `docker images | grep "^<none>" | awk "{print $3}"`

.PHONY: build bash bash2 prune killall rmi_untagged
