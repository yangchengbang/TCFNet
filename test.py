import argparse
import os
from dataset import Dataset
import torch
from torchvision import transforms
import transform
from torch.utils import data
from model.TCFNet import Model
import numpy as np
import cv2

parser = argparse.ArgumentParser()
print(torch.cuda.is_available())
parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda

# test
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--num_thread', type=int, default=0)
parser.add_argument('--input_size', type=int, default=384)
parser.add_argument('--model_path', type=str, default='./result_tr/final_bone.pth')
parser.add_argument('--test_dataset', type=list, default=['DAVIS/', 'FBMS/', 'SegTrack-V2/', 'ViSal/', 'DAVSOD/', 'VOS/'])
parser.add_argument('--testsavefold', type=str, default='./result_te')

# Misc
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
config = parser.parse_args()

composed_transforms_te = transforms.Compose([
    transform.FixedResize(size=(config.input_size, config.input_size)),
    transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    transform.ToTensor()])

dataset = Dataset(datasets=config.test_dataset, transform=composed_transforms_te, mode='test')
test_loader = data.DataLoader(dataset, batch_size=config.test_batch_size, num_workers=config.num_thread,
                              drop_last=True, shuffle=False)

print('mode: {}'.format(config.mode))
print('------------------------------------------')
net_bone = Model(3, mode=config.mode)
name = "TCFNet"
if config.cuda:
    net_bone = net_bone.cuda()
assert (config.model_path != ''), ('Test mode, please import pretrained model path!')
assert (os.path.exists(config.model_path)), ('please import correct pretrained model path!')
print('load model……all checkpoints')

net_bone.load_pretrain_model(config.model_path)
net_bone.eval()

if not os.path.exists(config.testsavefold):
    os.makedirs(config.testsavefold)

for i, data_batch in enumerate(test_loader):
    print("progress {}/{}\n".format(i + 1, len(test_loader)))
    image, flow, depth, name, split, size = data_batch['image'], data_batch['flow'], data_batch['depth'], \
                                            data_batch['name'], data_batch['split'], data_batch['size']
    dataset = data_batch['dataset']

    if config.cuda:
        image, flow, depth = image.cuda(), flow.cuda(), depth.cuda()
    with torch.no_grad():
        out1u, out2u, out1r, out2r, out3r, out4r, out5r, course_img, course_flowmap, course_depthmap, out_atfifm_layer1, out_atfifm_layer2, out_atfifm_layer3, out_atfifm_layer4, out_atfifm_layer5, out_deco1_layer1, out_deco1_layer2, out_deco1_layer3, out_deco1_layer4, out_deco1_layer5 = net_bone(image, flow, depth)
        for i in range(config.test_batch_size):
            presavefold = os.path.join(config.testsavefold,'pre_out2u/', dataset[i], split[i])
            if not os.path.exists(presavefold):
                os.makedirs(presavefold)
            pre1 = torch.nn.Sigmoid()(out2u[i])
            pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
            pre1 = np.squeeze(pre1.cpu().data.numpy()) * 255
            pre1 = cv2.resize(pre1, (int(size[0][1]), int(size[0][0])))
            cv2.imwrite(presavefold + '/' + name[i], pre1)

