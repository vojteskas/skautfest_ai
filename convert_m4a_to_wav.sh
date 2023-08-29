#!/bin/bash 

for file in ./dataset/*/*.m4a;
do
    name="${file//m4a/wav}"
    echo "$name"
    ffmpeg -i "$file" "${name}" 2>/dev/null

    # rm "$file"
done