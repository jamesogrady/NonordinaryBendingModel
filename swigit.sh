#!/usr/bin/env bash

#The MIT License (MIT)

#Copyright (c) 2015 James O'Grady

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

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
