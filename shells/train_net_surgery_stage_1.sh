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
caffe_dir=$HOME/caffe-fcn
$caffe_dir/python/draw_net.py train_test_stage_1.prototxt net_stage_1.png
echo 'start learning' $1

$caffe_dir/build/tools/caffe train -solver $PWD/solver_stage_1.prototxt