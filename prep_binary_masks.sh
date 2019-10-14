#!/bin/sh
ldconfig
env PYTHONPATH=.:./face-alignment/ python3 prep_binary_masks.py $@
