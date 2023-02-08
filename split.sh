input="/home/brain/Downloads/imagesets/ImageSets/val.txt"
mkdir -p "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/training/image_2/"
mkdir -p "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/training/velodyne/"
mkdir -p "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/training/label_2/"

mkdir -p "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/testing/image_2/"
mkdir -p "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/testing/velodyne/"
mkdir -p "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/testing/label_2/"

while IFS= read -r line
do
  echo "Copying ${line} ... "
#  cp "/home/brain/Oddy/Dataset/training/image_2/${line}.png" "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/training/image_2/${line}.png"
#  cp "/home/brain/Oddy/Dataset/training/velodyne/${line}.bin" "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/training/velodyne/${line}.bin"
#  cp "/home/brain/Oddy/Dataset/training/label_2/${line}.txt" "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/training/label_2/${line}.txt"
  
  cp "/home/brain/Oddy/Dataset/training/image_2/${line}.png" "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/testing/image_2/${line}.png"
  cp "/home/brain/Oddy/Dataset/training/velodyne/${line}.bin" "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/testing/velodyne/${line}.bin"
  cp "/home/brain/Oddy/Dataset/training/label_2/${line}.txt" "/home/brain/Oddy/Projects/VoxelNet-tensorflow/data/object/testing/label_2/${line}.txt"
done < "$input"
