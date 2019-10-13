#!/bin/sh
docker run -it -p 9999:8888 --rm -v $PWD:/tf -w /tf faceswap-gan $@

