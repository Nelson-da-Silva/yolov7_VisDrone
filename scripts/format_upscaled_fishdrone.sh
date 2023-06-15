#!/bin/bash
# To be run from the yolov7 directory e.g.
#   chmod /scripts/format_fishdrone.sh
#   ./scripts/format_fishdrone.sh

# Run the python script that downloads the Fishdrone dataset
python get_upscaled_fishdrone.py

dir="FishDrone"

# Merge the oversized training images together into one folder
mv $dir/train-600-images-2/* $dir/train-600/images/
mv $dir/train-830-images-2/* $dir/train-830/images/
mv $dir/train-830-images-3/* $dir/train-830/images/

# Delete the overflow folders so that they don't mess with the upcoming for loops
rm -r $dir/train-600-images-2
rm -r $dir/train-830-images-2
rm -r $dir/train-830-images-3

#   Create a folder containing all images of the varying focal lengths together
mkdir -p $dir/all/train/images
mkdir -p $dir/all/train/labels
mkdir -p $dir/all/train/annotations

mkdir -p $dir/all/test/images
mkdir -p $dir/all/test/labels
mkdir -p $dir/all/test/annotations

mkdir -p $dir/all/val/images
mkdir -p $dir/all/val/labels
mkdir -p $dir/all/val/annotations

# Rename while copying to the unified directory
for g in $dir/train-* # Iterate through train folders
do
    fl="$(cut -d'-' -f2 <<<$g)" # Focal length
    for f in $g/*/* # Iterate through both folders
    do
        type=$(basename $(dirname $f)) # image or label
        name=$(basename "$f") # Get name of file to be copied over
        cp -- "$f" "$dir/all/train/$type/f"$fl"_"$name # copy over with prefix
    done
done

for g in $dir/test-* # Iterate through test folders
do
    fl="$(cut -d'-' -f2 <<<$g)" # Focal length
    for f in $g/*/* # Iterate through both folders
    do
        type=$(basename $(dirname $f)) # image or label
        name=$(basename "$f") # Get name of file to be copied over
        cp -- "$f" "$dir/all/test/$type/f"$fl"_"$name # copy over with prefix
    done
done

for g in $dir/val-* # Iterate through val folders
do
    fl="$(cut -d'-' -f2 <<<$g)" # Focal length
    for f in $g/*/* # Iterate through both folders
    do
        type=$(basename $(dirname $f)) # image or label
        name=$(basename "$f") # Get name of file to be copied over
        cp -- "$f" "$dir/all/val/$type/f"$fl"_"$name # copy over with prefix
    done
done

# Needed three separate for loops rather than having them all in one to avoid
#   detecting the /all directory containing all labels combined