#!/bin/sh
docker run --runtime nvidia -it -p 9999:8888 --rm -v $PWD:/tf -v $PWD/cache:/root/.cache -w /tf faceswap-gan $@

