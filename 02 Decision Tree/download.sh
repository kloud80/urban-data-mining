URL=$1
ZIP_FILE=data.zip
TARGET_DIR=data/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d data/
rm $ZIP_FILE