# Make sure that caffe is on the python path:
caffe_root = '../../caffe-fcn/'  # this file is expected to be in {ssai_root}/scripts
import sys
import caffe
import argparse

sys.path.insert(0, caffe_root + 'python')
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', type=str)
parser.add_argument('--caffemodel', '-c', type=str)
args = parser.parse_args()

if args.mode == 'train':
    model_dir = 'VGG_PReLU_Roads_Mini_Net_Surgery_Train_and_Test'

if args.mode == 'test':
    model_dir = 'VGG_PReLU_Roads_Mini_Net_Surgery_Test_Only'

caffemodel = args.caffemodel
out_file = caffemodel.replace('.caffemodel', '_conv.caffemodel')

print "loading original net"
# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('./models/%s/predict.prototxt' %model_dir,
                '%s'%caffemodel,
                caffe.TEST)
params = ['fc33', 'fc36', 'fc39']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

print "loading fcn..."

# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('./models/%s/predict_conv.prototxt' %model_dir,
                          '%s'%caffemodel,
                          caffe.TEST)
params_full_conv = ['fc33-conv', 'fc36-conv', 'fc39-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)    

print "looping through conv params and doing transplant..."

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save('%s' %out_file)