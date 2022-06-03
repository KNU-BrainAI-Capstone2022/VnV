#!/bin/bash
if [ $# -ne 2 ]; then
    echo "Use : bash setup_cityscapes.sh [data_dir] [dataset]"
fi

if [ -z $1 ]; then
    mkdir $1
fi

cd $1

if [ $2 == 'voc2012' ]; then
    echo "Setup Pascal VOC 2012 Dataset"
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar -xvf VOCtrainval_11-May-2012.tar
    rm VOCtrainval_11-May-2012.tar
elif [ $2 == 'cityscapes' ]; then
    echo "Setup Cityscapes Dataset"
    echo "You must have an account at https://www.cityscapes-dataset.com/"
    read -p "Username : " name
    read -sp "Password : " passwd

    wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=$name&password=$passwd&submit=Login" https://www.cityscapes-dataset.com/login/
    # Download JPEG Images
    wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
    unzip leftImg8bit_trainvaltest.zip
    if [ -e 'index.html' ]; then
        rm index.html
    fi
    if [ -e 'license.txt' ]; then
        rm license.txt
    fi
    if [ -e 'README' ]; then
        rm README
    fi
    rm leftImg8bit_trainvaltest.zip
    # Download Ground Truth
    wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
    unzip gtFine_trainvaltest.zip
    if [ -e 'index.html' ]; then
        rm index.html
    fi
    if [ -e 'license.txt' ]; then
        rm license.txt
    fi
    if [ -e 'README' ]; then
        rm README
    fi
    if [ -e 'cookies.txt' ]; then
        rm cookies.txt
    fi
    rm gtFine_trainvaltest.zip
elif [ $2 == 'cityscapes_pair' ]; then
    echo "Setup Cityscapes_pair Dataset"
    echo "pair image size : 256x512"
    if [ $(which python) ] && [ $(which pip) ]; then
        echo "If kaggle is not installed on the pip, the kaggle package is installed."
        echo -e "The kaggle.json file must be located under ~/.kaggle (based on Unix-based).\nOtherwise, obtain the token from the account on the kaggle homepage through Create New API Token and move it to the ~/.kaggle location."
        python -m pip install kaggle
        kaggle datasets download -d dansbecker/cityscapes-image-pairs
        unzip cityscapes-image-pairs.zip
        rm cityscapes-image-pairs.zip
        mv cityscapes_data cityscapes_pair
        cd cityscapes_pair
        rm -rf cityscapes_data
    else
        echo "You must have python and pip installed."
    fi
else
    echo "dataset choice : {voc2012,cityscapes,cityscapes_pair}"
fi