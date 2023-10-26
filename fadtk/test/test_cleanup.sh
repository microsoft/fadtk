#!/usr/bin/env bash

# Get script path's directory
src=$(dirname "$0")
echo "Cleaning up $src"

rm -rfv "$src/fad_scores"
rm -rfv "$src/comparison.csv"
rm -rfv "$src/samples/convert"
rm -rfv "$src/samples/embeddings"
