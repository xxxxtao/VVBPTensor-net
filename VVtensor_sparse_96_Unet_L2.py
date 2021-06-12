
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:30:30 2019

@author: wyb081@smu.edu.cn
"""

import torch
import numpy as np
import numpy.random as random
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
import pickle
import time
import os
import skimage
import copy
import scipy.io
import scipy.io as sio
import pdb
import  torchvision.models as models
from Model import networks_1 as networks
from Model import common
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

pdb.set_trace()

is_train = False     # True: Train, False: Test 
re_load = True       # True: Load the trained model 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
WaterAtValue = 0.0192
source_root_path = '/mnt/storage/VVtenosr-net/Data/'
target_root_path = '/mnt/storage/VVtenosr-net/Model_save/'
Train_reconinfo = {'patients': ['L096', 'L291','L109','L143','L192','L286','L333','L310'], 'SliceThickness': ['']}
Val_reconinfo = {'patients': ['L506','L067'], 'SliceThickness': ['']} # 
ResultFolders = ['Model_save', 'Loss_save', 'Optimizer_save', 'SinoIndices_save']


reload_mode = 'train'
batch_num = {'train': 200, 'val': 10, 'test': 1}
batch_size= {'train': 1, 'val': 1, 'test': 1}
is_lr_scheduler = False
filter_size = 3
filter_num = 64
padding_size = 1#(filter_size-1)/2
is_addnoise = False
I0 = 3e8
sigma = 0.01
net_id = 1
net_name = 'VVtensor_sparse_96_Unet_L2'
indices_name = 'VVtenser_96_fan'

for target_folder in ResultFolders:
    if not os.path.isdir(target_root_path + net_name + '/' + target_folder):
        os.makedirs(target_root_path + net_name + '/' + target_folder)

TestMode = 'test'

    
target_folder = 'result'
save_as_mat = True
if is_train is True:
    is_shuffle = True
else:
    is_shuffle = False
    if not os.path.isdir(target_root_path + net_name + '/' + target_folder):
        os.makedirs(target_root_path + net_name + '/' + target_folder)

gpu_id_end = [0]


geo = {'nVoxelX': 512, 'nVoxelY': 512, 
       'sVoxelX': 355.0208, 'sVoxelY': 355.0208, 
       'dVoxelX': 0.6934, 'dVoxelY':  0.6934, 
       'sino_views': 96, 'sparse_factor': 12,
       'nDetecU': 736, 'sDetecU': 0.6934*736,
       'dDetecU':  1.2858, 'DSD': 1085.6, 'DSO': 595.0,
       'offOriginX': 0, 'offOriginY': 0, 
       'offDetecU': 0,
       'start_angle': 0, 'end_angle': 360,
        'mode': 'fan', 
      }

def PixelIndexCal(geo):

    nx = geo['nVoxelX']
    ny = geo['nVoxelY']
    offset_x = geo['offOriginX']
    offset_y = geo['offOriginY']

    wx =  (nx+1)/2 + offset_x
    wy = (ny+1)/2 + offset_y
    is_arc = 1
    
    dx = geo['dVoxelX']
    dy = -geo['dVoxelX']
    dr = geo['dDetecU']
    offset_s = geo['offDetecU']
    na = geo['sino_views']
    orbit = geo['end_angle'] - geo['start_angle']
    orbit_start = geo['start_angle']
    nb =  geo['nDetecU']
    dso = geo['DSO']
    dsd = geo['DSD']
    source_offset = 0
    ds = geo['dDetecU']
    xc, yc = np.mgrid[ dx-wx*dx : (nx-wx)*dx+dx :dx, dy-wy*dy : (ny-wy)*dy+dy :dy]

    
    rr = np.sqrt(xc**2+yc**2)

    betas = np.radians(np.arange(0,na).reshape(na,1)/na * orbit + orbit_start)
    wb = (nb+1)/2 + offset_s


    sino_indices = torch.zeros(geo['nVoxelX']*geo['nVoxelY'], geo['sino_views'])
    Weight2 = torch.zeros(geo['nVoxelX']*geo['nVoxelY'], geo['sino_views'])

    for ia in range(na):
        print('Na!{}/{}'.format(ia+1,na))
        beta = betas[ia]
        d_loop = dso + xc * np.sin(beta) - yc * np.cos(beta) # dso - y_beta

        r_loop = xc * np.cos(beta) + yc * np.sin(beta) - source_offset # x_beta-roff

        
        if is_arc:
            sprime_ds = (dsd/ds) * np.arctan2(r_loop, d_loop)
            w2 = dsd**2 / (d_loop**2 + r_loop**2)           # [np] image weighting
        else:
            mag = dsd / d_loop
            sprime_ds = mag * r_loop / ds

        bb = sprime_ds + wb
        bb = bb + ia * nb - 1
        
        sino_indices[:,ia] = torch.from_numpy(bb).view(-1)
        Weight2[:,ia]  = torch.from_numpy(w2).view(-1)
        
    sino_indices = sino_indices.view(-1)
    Weight2 = Weight2.view(-1)

    
    return sino_indices,Weight2


if os.path.isfile(target_root_path+ net_name + "/SinoIndices_save/{}_indices.dat".format(indices_name)):
    print('Loading sinoIndices...')
    geo['indices'] = pickle.load(open(target_root_path+ net_name + "/SinoIndices_save/{}_indices.dat".format(indices_name), "rb"))
    geo['Weight2'] = pickle.load(open(target_root_path+ net_name + "/SinoIndices_save/{}_Weight2.dat".format(indices_name), "rb"))
    print('Done!')
else:
    print('Generating sinoIndices...')
    geo['indices'],geo['Weight2']  = PixelIndexCal(geo)
    f = open(target_root_path+ net_name + "/SinoIndices_save/{}_indices.dat".format(indices_name), "wb")
    pickle.dump(geo['indices'], f, True)
    f.close()
    f = open(target_root_path+ net_name + "/SinoIndices_save/{}_Weight2.dat".format(indices_name), "wb")
    pickle.dump(geo['Weight2'], f, True)
    f.close()

    print('Done!')


if use_cuda:
    geo['indices'] = geo['indices']
    geo['il'] = torch.floor(geo['indices'])
    geo['wr'] = geo['indices'] - geo['il']
    
    geo['indices'] = geo['indices'].cuda(gpu_id_end[0])
    geo['wr'] = geo['wr'].cuda(gpu_id_end[0])
    geo['il'] = geo['il'].type(torch.LongTensor).cuda(gpu_id_end[0])
    geo['Weight2'] = geo['Weight2'].cuda(gpu_id_end[0])


class Backprojected(nn.Module):
    def __init__(self, geo, bias=True):
        super(Backprojected, self).__init__()
        self.geo = geo
        

    def forward(self, input):
        
        input = input.view(-1, self.geo['sino_views']*self.geo['nDetecU'])
        
        input_ = ((1 -geo['wr']) * torch.index_select( input, 1, self.geo['il'] ) + geo['wr'] *  torch.index_select( input, 1, self.geo['il']+1 )) * self.geo['Weight2'] 
        
        input_ = input_.view(-1, self.geo['nVoxelX'], self.geo['nVoxelY'], self.geo['sino_views'])
        input_, _ = input_.sort(3,descending=False)


        return input_



class VVtenosr_Unet(nn.Module):
    def __init__(self):
        super(VVtenosr_Unet, self).__init__()
        
        self.backprojected = Backprojected(geo)
        self.DeepVVBP_Unt = networks.FBPCONVNet_big_drop(input_nc=128, NFS=128,p=0.1).cuda(gpu_id_end[0])
        self.rec_criterion = nn.MSELoss().cuda(gpu_id_end[0])

        act = nn.ReLU(True)

        VV_Compress1 =  [common.BBlock(common.default_conv, geo['sino_views'], 128, 3, act=act)]
        VV_Compress2 =  [common.BBlock(common.default_conv, 128, 128, 3, act=act)]
        
        output_ly =  [common.default_conv( 128, 1, 3)]



        self.VV_Compress1 = nn.Sequential(*VV_Compress1).cuda(gpu_id_end[0])
        self.VV_Compress2 = nn.Sequential(*VV_Compress2).cuda(gpu_id_end[0])

    
        self.output_ly = nn.Sequential(*output_ly).cuda(gpu_id_end[0])

    def forward(self, Image_LD, sino_sparse, Image_HD):
   
        
        x =  self.backprojected(sino_sparse)
        x = x.permute(0,3,1,2).contiguous()
        Image_LD = torch.sum(x,1)*np.pi*20/geo['sino_views']

        x = self.VV_Compress1(x) 
        x = self.VV_Compress2(x) 
    
        x = self.DeepVVBP_Unt(x)
        x = self.output_ly(x)


        Loss_rec = self.rec_criterion(x, Image_HD) 


        return x, Loss_rec, Image_LD

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        return torch.from_numpy(image).type(torch.FloatTensor)

class TrainDicmDataset(Dataset):
    def __init__(self, root_dir, reconinfo, geo, trf_op=None):
        
        self.Raw_data_paths = [glob.glob( root_dir +'{}/*'.format(x) ) for x in reconinfo['patients']]
        self.Raw_data_paths = [x for j in self.Raw_data_paths for x in j]
        self.trf_op = trf_op
        self.geo = geo

        
    def __len__(self):
        return len(self.Raw_data_paths)
    
    def __getitem__(self, idx):
        # Label
        try:
            image_path = self.Raw_data_paths[idx] + '/ND/Data.mat'
            data_HD = sio.loadmat(image_path) 
        except ValueError as a:
            
            print('*******************************************************************************')
            print ('Exception: ', a)
            print('*******************************************************************************')
            print(self.Raw_data_paths[idx])

            try:
                idx = random.randint(0,len(self.Raw_data_paths), size=([1]))[0]
                image_path = self.Raw_data_paths[idx] + '/ND/Data.mat'
                data_HD = sio.loadmat(image_path) 
            except ValueError as a:
                print('*******************************************************************************')
                print ('Exception: ', a)
                print('*******************************************************************************')
                print(self.Raw_data_paths[idx])

                try:
                    idx = random.randint(0,len(self.Raw_data_paths), size=([1]))[0]
                    image_path = self.Raw_data_paths[idx] + '/ND/Data.mat'
                    data_HD = sio.loadmat(image_path) 
                except ValueError as a:
                    print('*******************************************************************************')
                    print ('Exception: ', a)
                    print('*******************************************************************************')
                    print(self.Raw_data_paths[idx])

                    try:
                        idx = random.randint(0,len(self.Raw_data_paths), size=([1]))[0]
                        image_path = self.Raw_data_paths[idx] + '/ND/Data.mat'
                        data_HD = sio.loadmat(image_path) 

                    except ValueError as a:
                        print('*******************************************************************************')
                        print ('Exception: ', a)
                        print('*******************************************************************************')
                        print(self.Raw_data_paths[idx])
                        idx = random.randint(0,len(self.Raw_data_paths), size=([1]))[0]
                        image_path = self.Raw_data_paths[idx] + '/ND/Data.mat'
                        data_HD = sio.loadmat(image_path) 
                
        try:
            image_path_LD = self.Raw_data_paths[idx] + '/ND/Data.mat'
            data_LD = sio.loadmat(image_path_LD) 
        except ValueError as a:
            try:
                image_path_LD = self.Raw_data_paths[idx] + '/ND/Data.mat'
                data_LD = sio.loadmat(image_path_LD)
            except ValueError as a:
                image_path_LD = self.Raw_data_paths[idx+1] + '/ND/Data.mat'
                data_LD = sio.loadmat(image_path_LD)

        Image_HD = data_HD['Image'] * 20
        sino_sparse = np.transpose(data_LD['SinoFiltered'][:,::geo['sparse_factor']])
       
        Image_LD = data_HD['Image'] * 20
        
        del data_HD,data_LD
        #pdb.set_trace()

        image_path_split =  image_path.split('/')
        name = image_path_split[-4]+'_'+image_path_split[-3]

        random_list = [ToTensor()]
        transform = transforms.Compose(random_list)
        Image_HD = transform(Image_HD)
        Image_LD = transform(Image_LD)
        sino_sparse = transform(sino_sparse)
        return Image_HD, Image_LD, sino_sparse, name


class TrainDataSet(Dataset):
    def __init__(self, root_dir, reconinfo, geo, pre_trans_img=None, post_trans_img=None, post_trans_sino=None, addnoise=None):
        
        self.imgset = TrainDicmDataset(root_dir, reconinfo, geo, pre_trans_img)
        self.addnoise = transforms.Compose(addnoise) if addnoise is not None else None
        self.post_trans_img = transforms.Compose(post_trans_img) if post_trans_img is not None else None
        self.post_trans_sino = transforms.Compose(post_trans_sino) if post_trans_sino is not None else None
        
    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx):
        Image_HD, Image_LD, sino_sparse, name = self.imgset[idx]
   
    
        sample = {'Image_HD': Image_HD, 'Image_LD': Image_LD, 'name': name, 'sino_sparse':sino_sparse}
        sample['Image_HD'].unsqueeze_(0)
        sample['Image_LD'].unsqueeze_(0)
        sample['sino_sparse'].unsqueeze_(0)
        return sample


pre_trans_img = None
addnoise = [AddNoise(I0, sigma)] if is_addnoise is True else None
post_trans_img = None
post_trans_sino = None
datasets = []
datasets = {'train': TrainDataSet(source_root_path, Train_reconinfo, geo, pre_trans_img, post_trans_img, post_trans_sino, addnoise),
            'val': TrainDataSet(source_root_path, Val_reconinfo, geo, pre_trans_img, post_trans_img, post_trans_sino, addnoise),
            'test': TrainDataSet(source_root_path, Val_reconinfo, geo, pre_trans_img, post_trans_img, post_trans_sino, addnoise)}

kwargs = {'num_workers': 4, 'pin_memory': True}
data = datasets['test'].__getitem__(0)


dataloaders = {x: DataLoader(datasets[x], batch_size[x], shuffle=is_shuffle, **kwargs) for x in ['train', 'val', 'test']}

dataset_sizes = {x: batch_num[x]*batch_size[x] for x in ['train', 'val', 'test']}

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def train_model(model, optimizer, criterion=None, scheduler=None, min_loss=None, pre_losses=None, num_epochs=25):
    since = time.time()

    if min_loss is None:
        min_loss = {x:1.0 for x in ['train', 'val']}
        
    losses = {x: torch.zeros(num_epochs, batch_num[x]) for x in ['train', 'val']}
    
    if pre_losses is not None:
        min_dim = {x: min(losses[x].size(1), pre_losses[x].size(1)) for x in ['train', 'val']}
        
    epoch_loss = {x: 0.0 for x in ['train', 'val']}

    print( 'lr={:.10f}'.format(optimizer.param_groups[0]['lr']))
    for epoch in range(num_epochs):
        if (epoch == 100) or (epoch == 300)  :
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.2
            print( 'lr={:.10f}'.format(optimizer.param_groups[0]['lr']))

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase, 'val'
        for phase in ['train','val']: 
            if phase == 'train':
                if scheduler is not None:
                    print('-' * 10)#scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
  
            
            # Iterate over data.
            for i_batch, Sample in enumerate(dataloaders[phase]):
                
                
                if i_batch == batch_num[phase]:
                    break
                Len_data = datasets[phase].__len__()

                
               
                Image_HD = Sample['Image_HD']
                Image_LD = Sample['Image_LD']
                sino_sparse = Sample['sino_sparse']
                
                if use_cuda:  # wrap them in Variable
                    Image_HD = Variable(Image_HD).cuda(gpu_id_end[0])
                    Image_LD = Variable(Image_LD).cuda(gpu_id_end[0], async=True)
                    sino_sparse = Variable(sino_sparse).cuda(gpu_id_end[0], async=True)
                else:
                    Image_HD = Variable(Image_HD)
                    Image_LD = Variable(Image_LD)
                    
                
                # zero the parameter gradients
                optimizer.zero_grad()
            
                # forward
                outputs,  loss, Image_LD = model(Image_LD, sino_sparse, Image_HD)
                #pdb.set_trace()
                if (phase == 'val') * (i_batch==0):
                    
                    scipy.io.savemat(target_root_path + net_name + '/val.mat',  mdict = {'Image_HD':Image_HD.cpu().data.numpy(),'Image_LD':Image_LD.cpu().data.numpy(),'outputs': outputs.cpu().data.numpy() } ) 
                
                del Image_HD, Image_LD, Sample, outputs, sino_sparse
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                   
                    optimizer.step()

                if 0 == (i_batch%1):
                    print('{}:, {}, subIter {}/{}, subLoss: {:.8f}'.format(net_name,phase, i_batch, batch_num[phase], loss.data.item()))
                # statistics
                losses[phase][epoch, i_batch] = loss.data.item()
                
                running_loss += loss.data.item() * batch_size[phase]
                del loss


            epoch_loss[phase] = running_loss / (dataset_sizes[phase])

            print('{} Loss: {:.8f}'.format(phase, epoch_loss[phase]))
            
            if phase == 'val':
                print("Train / Val: {:.8f}".format(epoch_loss['train']/epoch_loss['val']))
               
            

            #deep copy the model
            if 0 == (epoch%1):
                if 1: # epoch_loss[phase] < min_loss[phase]:
                    min_loss[phase] = epoch_loss[phase]
                    f = open(target_root_path + net_name + "/Loss_save/min_loss_{}_{}.dat".format(net_id, net_name), "wb")
                    pickle.dump(min_loss, f, True)
                    f.close()
                    torch.save(model.state_dict(), 
                            target_root_path + net_name + "/Model_save/best_{}_model_params_{}_{}.pkl".format(phase, net_id, net_name))
                    if phase is 'train':
                        torch.save(optimizer.state_dict(),
                                target_root_path + net_name + "/Optimizer_save/optimizer_{}_{}.pkl".format(net_id, net_name))
                        
                

        if pre_losses is None:
            tmp_losses = {x: losses[x][:epoch+1] for x in ['train', 'val']}
        else:
            tmp_losses = {x: torch.cat((pre_losses[x][:,:min_dim[x]], losses[x][:epoch+1,:min_dim[x]]), 0) for x in ['train', 'val']}
        f = open(target_root_path + net_name + "/Loss_save/losses_{}_{}.dat".format(net_id, net_name), "wb")
        pickle.dump(tmp_losses, f, True)
        f.close()

        
        print( 'lr={:.10f}'.format(optimizer.param_groups[0]['lr']))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Minimun train loss: {:5f}'.format(min_loss['train']))

def test_model(model, criterion=None):
    for i_batch, Sample in enumerate(dataloaders[TestMode]):
        print('processing batch_{}...'.format(i_batch))

        
        Image_HD = Sample['Image_HD']
        Image_LD = Sample['Image_LD']
        sino_sparse = Sample['sino_sparse']
                
        if use_cuda:  # wrap them in Variable
            Image_HD = Variable(Image_HD).cuda(gpu_id_end[0])
            Image_LD = Variable(Image_LD).cuda(gpu_id_end[0], async=True)
            sino_sparse = Variable(sino_sparse).cuda(gpu_id_end[0], async=True)
        else:
            Image_HD = Variable(Image_HD)
            Image_LD = Variable(Image_LD)

        outputs, loss, Image_LD = model(Image_LD, sino_sparse, Image_HD)

    
        Sample['Image_LD'] = Image_LD/20

        Sample['Image_HD'] = Image_HD/20

        Sample['output'] = outputs/20
        Sample['loss'] = loss
        data_name = ''.join(Sample['name'])
        Sample.pop('name')
        
        del outputs, loss, Image_LD
        
        
        #pdb.set_trace()
        data_save = {key: value.cpu().data.numpy() for key, value in Sample.items()}
        scipy.io.savemat(target_root_path + net_name  + '/result/{}.mat'.format(data_name), mdict = data_save)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1: #or classname.find('Backprojected') != -1:
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
        
        
Network_im = VVtenosr_Unet()

if re_load is False:
    Network_im.apply(weights_init)
    min_loss = None
    pre_losses = None
else:
    epoch_reload_path = target_root_path + net_name + '/Model_save/best_{}_model_params_{}_{}.pkl'.format(reload_mode, net_id, net_name)
    if os.path.isfile(epoch_reload_path):
        print('reloading previously trained network...')
        checkpoint = torch.load(epoch_reload_path, map_location = lambda storage, loc: storage)
        model_dict = Network_im.state_dict()
        checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        Network_im.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        print('done!')
    else:
        Network_im.apply(weights_init)
        
    min_loss_vggath = target_root_path + net_name + "/Loss_save/min_loss_{}_{}.dat".format(net_id, net_name)
    min_loss = pickle.load(open(min_loss_vggath, "rb")) if os.path.isfile(min_loss_vggath) else None
        
    pre_losses_path = target_root_path + net_name + "/Loss_save/losses_{}_{}.dat".format(net_id, net_name)
    pre_losses = pickle.load(open(pre_losses_path, "rb")) if os.path.isfile(pre_losses_path) else None
    
    
criterion =None

optimizer_ft = optim.RMSprop(Network_im.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.0000)
#optimizer_ft = optim.Adam(Network_im.parameters(), lr=1e-5, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)
if re_load is True:
    optimizer_reload_path = target_root_path+ net_name + "/Optimizer_save/optimizer_{}_{}.pkl".format(net_id, net_name)
    if os.path.isfile(optimizer_reload_path):
        print('reloading previous optimizer...')
        checkpoint = torch.load(optimizer_reload_path, map_location = lambda storage, loc: storage)
        optimizer_ft.load_state_dict(checkpoint) 
        del checkpoint
        torch.cuda.empty_cache()
        print('done!')



exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=300, gamma=0.5) if is_lr_scheduler else None

if is_train is True:
    train_model(Network_im, optimizer_ft, criterion, exp_lr_scheduler, min_loss, pre_losses, num_epochs=400)
else:
    Network_im.eval()
    test_model(Network_im, criterion)





