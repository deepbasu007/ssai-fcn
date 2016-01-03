#! /bin/bash
fn=`date +"%Y-%m-%d_%I-%M-%S"`
if [ ! -d results ]; then
    mkdir results
fi
cd results
fpath=$1

modelname="${fpath##*/}"
echo $modelname
dname=$modelname'_'$fn
echo $dname
mkdir $dname
cd $dname
mkdir snapshots
cp ../../models/$modelname/*.prototxt ./
caffe_dir=$HOME/Downloads/caffe-fcn
$caffe_dir/python/draw_net.py train_test.prototxt net.png

echo 'start learning' $1

$caffe_dir/build/tools/caffe train -solver $PWD/solver.prototxt