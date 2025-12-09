import argparse
from dataset import Dataset
from torchvision import transforms
import transform
from torch.utils import data
import torch
from collections import OrderedDict
from model.TCFNet import Model
import os
import numpy as np
import IOU
import datetime
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

p = OrderedDict()
p['lr_bone'] = 1e-4  # Learning rate
p['lr_branch'] = 1e-3
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [9, 20]
showEvery = 50

CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)

def structure_loss(pred, mask):
    bce = CE(pred, mask)
    iou = IOU(torch.nn.Sigmoid()(pred), mask)
    return bce + iou
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
print(torch.cuda.is_available())

parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda

# train
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--epoch_save', type=int, default=5)
parser.add_argument('--save_fold', type=str, default='./result_tr')  # 训练过程中输出的保存路径
parser.add_argument('--input_size', type=int, default=384)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_thread', type=int, default=0)

parser.add_argument('--model_path', type=str, default='')

# Misc
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
config = parser.parse_args()

if not os.path.exists("%s" % (config.save_fold)):
    os.mkdir("%s" % (config.save_fold))

if __name__ == '__main__':
    set_seed(1024)

    composed_transforms_ts = transforms.Compose([
        transform.RandomFlip(),
        transform.RandomRotate(),
        transform.colorEnhance(),
        transform.randomPeper(),
        transform.FixedResize(size=(config.input_size, config.input_size)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])
    dataset_train = Dataset(datasets=['DAVIS/', 'DAVSOD/', 'FBMS/', 'DUTS'], transform=composed_transforms_ts, mode='train')

    dataloader = data.DataLoader(dataset_train, batch_size=config.batch_size, num_workers=config.num_thread, drop_last=True, shuffle=True)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_train), len(dataloader)))


    net_bone = Model(3, mode=config.mode)
    if config.cuda:
        net_bone = net_bone.cuda()

    base, head = [], []
    for name, param in net_bone.named_parameters():
        if 'rgb_bkbone' in name or 'flow_bkbone' in name or 'depth_bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer_bone = torch.optim.SGD([{'params': base}, {'params': head}], lr=p['lr_bone'], momentum=p['momentum'],
                                     weight_decay=p['wd'], nesterov=True)

    optimizer_bone.zero_grad()

    iter_num = len(dataloader)
    for epoch in range(config.epoch):
        loss_all = 0
        optimizer_bone.param_groups[0]['lr'] = p['lr_bone']
        optimizer_bone.param_groups[1]['lr'] = p['lr_branch']
        net_bone.zero_grad()

        net_bone.train()

        for i, data_batch in enumerate(dataloader):
            image, label, flow, depth = data_batch['image'], data_batch['label'], data_batch['flow'], data_batch['depth']
            if image.size()[2:] != label.size()[2:]:
                print("Skip this batch")
                continue
            if config.cuda:
                image, label, flow, depth = image.cuda(), label.cuda(), flow.cuda(), depth.cuda()

            out1u, out2u, out1r, out2r, out3r, out4r, out5r, course_img, course_flowmap, course_depthmap, out_atfifm_layer1, out_atfifm_layer2, out_atfifm_layer3, out_atfifm_layer4, out_atfifm_layer5, out_deco1_layer1, out_deco1_layer2, out_deco1_layer3, out_deco1_layer4, out_deco1_layer5 = net_bone(image, flow, depth)

            loss1u = structure_loss(out1u, label)
            loss2u = structure_loss(out2u, label)

            loss1r = structure_loss(out1r, label)
            loss2r = structure_loss(out2r, label)
            loss3r = structure_loss(out3r, label)
            loss4r = structure_loss(out4r, label)
            loss5r = structure_loss(out5r, label)

            loss_img_feature = structure_loss(course_img, label)
            loss_flow_feature = structure_loss(course_flowmap, label)
            loss_depth_feature = structure_loss(course_depthmap, label)

            loss = loss1u + loss2u + loss1r / 2 + loss2r / 4 + loss3r / 8 + loss4r / 16 + loss5r / 32 + loss_img_feature * 0.5 + loss_flow_feature * 0.2 + loss_depth_feature * 0.3

            optimizer_bone.zero_grad()
            loss.backward()
            optimizer_bone.step()
            loss_all += loss.data

            if i % showEvery == 0:
                print(
                    '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss1 : %10.4f  || sum : %10.4f' % (
                        datetime.datetime.now(), epoch, config.epoch, i, iter_num,
                        loss2u.data, loss_all / (i + 1)))
                print('Learning rate: ' + str(optimizer_bone.param_groups[0]['lr']))

        if (epoch + 1) % config.epoch_save == 0:
            torch.save(net_bone.state_dict(),
                       '%s/epoch_%d_bone.pth' % (config.save_fold, epoch + 1))

        if epoch in lr_decay_epoch:
            p['lr_bone'] = p['lr_bone'] * 0.2
            p['lr_branch'] = p['lr_branch'] * 0.2

    torch.save(net_bone.state_dict(), '%s/final_bone.pth' % config.save_fold)
