FILE=$1
echo "Specified [$FILE]"
URL=[$FILE]
ZIP_FILE=./data.zip
TARGET_DIR=./data/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE