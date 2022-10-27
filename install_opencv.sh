# install opencv
apt-get install cmake gcc g++ -y

# apt purge libopencv* python-opencv
# apt autoremove

# apt install ffmpeg

# apt-get install -y openjdk-8-jdk
# export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
apt-get -y libglew-dev libtiff5-dev zlib1g-dev libjpeg-dev \
libpng12-dev libjasper-dev libavcodec-dev libavformat-dev \
libavutil-dev libpostproc-dev libswscale-dev libeigen3-dev \
libtbb-dev libgtk2.0-dev pkg-config
apt update -y

apt-get install -y python3-dev python3-numpy python3-py python3-pytest
# apt-get install -y build-essential cmake git pkg-config dkms \
# freeglut3 freeglut3-dev libxi-dev libxmu-dev \
# libavcodec-dev libavformat-dev libswscale-dev libv4l-dev v4l-utils libavutil-dev \
# libgstreaer-plugins-base1.0-dev libgstreamer1.0-dev libgtk2.0-dev libgtk-3-dev \
# libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev \
# libxvidcore-dev libx264-dev libxine2-dev libjpeg-dev libpng-dev libtiff-dev \
# gfortran openexr libatlas-base-dev python3-dev python3-numpy \
# libgl1-mesa-dri libeigen3-dev pyflakes pylint
# libfreetype6-dev libharfbuzz-dev # for korean on opencv
# install pip
# pip3 install --upgrade pip
# pip3 install --upgrade setuptools pip
# pip3 install numpy 
# pip3 install flake8
# pip3 install pylint

OPENCV_VER="4.5.1"
# download opencv
cd ../
if [ -d 'opencv' ]; then
    echo ''
else
    mkdir opencv
fi
cd opencv
if [ -e 'opencv.zip' ]; then
    echo "opencv.zip is exists."
else
    wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VER.zip
fi

if [ -e 'opencv_contrib.zip' ]; then
    echo "opencv_contrib.zip is exists."
else
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/$OPENCV_VER.zip
fi

if [ -d "opencv-$OPENCV_VER" ]; then
    echo "opencv directory is exists"
else
    unzip opencv.zip
fi

if [ -d "opencv_contrib-$OPENCV_VER" ]; then
    echo "opencv_contrib directory is exists"
else
    unzip opencv_contrib.zip
fi

# lastest version
# git clone https://github.com/opencv/opencv.git
# git clone https://github.com/opencv/opencv_contrib.git
cd ./opencv-4.5.1
if [ -d 'build' ]; then
    echo ''
else
    mkdir build
fi
cd build
pwd

# GPU compute capability = ARCH_BIN
# cmake -D CMAKE_BUILD_TYPE=RELEASE \
# 	-D CMAKE_INSTALL_PREFIX=/usr/local \
#     -D BUILD_opencv_python2=OFF \
#     -D BUILD_opencv_python3=ON \
# 	-D INSTALL_PYTHON_EXAMPLES=ON \
# 	-D WITH_TBB=ON \
# 	-D OPENCV_ENABLE_NONFREE=ON \
# 	-D WITH_CUDA=ON \
# 	-D WITH_CUDNN=ON \
# 	-D OPENCV_DNN_CUDA=ON \
# 	-D ENABLE_FAST_MATH=1 \
# 	-D CUDA_FAST_MATH=1 \
# 	-D CUDA_ARCH_BIN=8.6 \
# 	-D WITH_CUBLAS=1 \
# 	-D WITH_OPENGL=ON \
#     -D WITH_VTK=OFF \
#     -D BUILD_opencv_viz=OFF \
#     -D OPENCV_GENERATE_PKGCONFIG=ON \
#     -D WITH_GSTREAMER=OFF \
# 	-D OpenGL_GL_PREFERENCE=LEGACY \
# 	-D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.5.1/modules \
# 	-D BUILD_EXAMPLES=ON \
#     -DPYTHON_DEFAULT_EXECUTABLE=$(which python3) \
#     -DBUILD_opencv_gapi:BOOL=OFF \
#     -DBUILD_PERF_TESTS:BOOL=OFF ..
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=ON \
    -D BUILD_DOCS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PACKAGE=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_TBB=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D WITH_CUDA=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_CUFFT=ON \
    -D WITH_NVCUVID=ON \
    -D WITH_IPP=OFF \
    -D WITH_V4L=ON \
    -D WITH_1394=OFF \
    -D WITH_GTK=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_EIGEN=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D BUILD_JAVA=OFF \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D OPENCV_SKIP_PYTHON_LOADER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.1/modules \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=8.6 \
    -D CUDA_ARCH_PTX=8.6 \
    -D CUDNN_INCLUDE_DIR=/usr/local/cuda/include  .. 

make -j$(nproc)

make install

/bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
ldconfig