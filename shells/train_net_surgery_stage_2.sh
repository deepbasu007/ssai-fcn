#! /bin/bash

# Usage: bash shells/train_net_surgery_stage_2.py <model_dir> <snapshot.caffemodel>

fn=`date +"%Y-%m-%d_%I-%M-%S"`
if [ ! -d results ]; then
    mkdir results
fi
cd results
fpath=$1
snapshot=$2

modelname="${fpath##*/}"
echo $modelname
dname=$modelname'_'$fn
echo $dname
mkdir $dname
cd $dname
mkdir snapshots
cp ../../models/$modelname/*.prototxt ./
caffe_dir=$HOME/caffe-fcn
$caffe_dir/python/draw_net.py train_test_stage_2.prototxt net_stage_2.png

ssai_dir=$HOME/ssai-fcn

echo 'start learning' $1

$caffe_dir/build/tools/caffe train -solver $PWD/solver_stage_2.prototxt  -weights $ssai_dir/$snapshot
