#!/usr/bin/env bash


PYTHONPATH=/share/apps/python/2.7.3

BASENAME=$1

swig -python $BASENAME.i

if [ "${2:-nodebug}" == "debug" ];then
    gcc -fPIC -g3 -c $BASENAME*.c \
        -I/usr/local/lib/python2.7/site-packages/numpy/core/include \
        -I/usr/local/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
        -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk/usr/include
elif [ "${2:-nodebug}" == "vectorize" ];then
    gcc -fPIC -ftree-vectorize -ftree-vectorizer-verbose=1 -O3 -c $BASENAME*.c \
        -I/usr/local/lib/python2.7/site-packages/numpy/core/include \
        -I/usr/local/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
        -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk/usr/include
else
    gcc -fPIC -O3 -c $BASENAME*.c \
        -I/usr/local/lib/python2.7/site-packages/numpy/core/include \
        -I/usr/local/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
        -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk/usr/include
fi

gcc -shared $BASENAME*.o -o _$BASENAME.so \
    -L/usr/local/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config \
    -ldl \
    -framework CoreFoundation \
    -lpython2.7




rm -f $BASENAME*.o
