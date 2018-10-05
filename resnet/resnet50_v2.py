from mxnet.gluon.model_zoo import vision
from mxnet.gluon.utils import download
from mxnet import image
from mxnet import nd
import matplotlib.pyplot as plt
import sys

net = vision.resnet50_v2(pretrained=True)
with open('./synset.txt', 'r', encoding='utf8') as f:
    text_labels = [l[9:].strip() for l in f]

def transform(data):
    data = data.transpose((2,0,1)).expand_dims(axis=0)
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std

def classify(image_file_path):
    x = image.imread(image_file_path)
    x = image.resize_short(x, 256)
    x, _ = image.center_crop(x, (224,224))
    plt.imshow(x.asnumpy())
    plt.show()
    prob = net(transform(x)).softmax()
    idx = prob.topk(k=5)[0]
    print('  prob  |  name')
    print('  ------------------')
    for i in idx:
        i = int(i.asscalar())
        print('  %.3f | %s' % (prob[0,i].asscalar(), text_labels[i]))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        classify(sys.argv[1])