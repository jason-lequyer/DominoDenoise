#Copyright 2021, Jason Lequyer and Laurence Pelletier, All rights reserved.
#Sinai Health SystemLunenfeld-Tanenbaum Research Institute
#600 University Avenue, Room 1070
#Toronto, ON, M5G 1X5, Canada
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tifffile import imread, imwrite
import sys
import torch.utils.data as utils_data
import numpy as np
from pathlib import Path
import time
from scipy.sparse import csr_matrix, csc_matrix, vstack, hstack, coo_matrix
import scipy.sparse.linalg as spl
import math
import scipy.sparse.csgraph as ssc


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        #####Yize's fixes
        self.multi_channel = True
        self.return_mask = True
        
        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output






if __name__ == "__main__":
    functions = [lambda x: x, lambda x: np.rot90(x), lambda x: np.rot90(np.rot90(x)), lambda x: np.rot90(np.rot90(np.rot90(x))), lambda x: np.flipud(x), lambda x: np.fliplr(x), lambda x: np.rot90(np.fliplr(x)), lambda x: np.fliplr(np.rot90(x))]
    ifunctions = [lambda x: x, lambda x: np.rot90(x,-1), lambda x: np.rot90(np.rot90(x,-1),-1), lambda x: np.rot90(np.rot90(np.rot90(x,-1),-1),-1), lambda x: np.flipud(x), lambda x: np.fliplr(x), lambda x: np.rot90(np.fliplr(x)), lambda x: np.fliplr(np.rot90(x))]
    folder = sys.argv[1]
    outfolder = folder+'_DD'
    Path(outfolder).mkdir(exist_ok=True)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    


    class TwoCon(nn.Module):
    
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = PartialConv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
            self.conv2 = PartialConv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
            self.nonlinear1 = nn.LeakyReLU(0.1)
            self.nonlinear2 = nn.LeakyReLU(0.1)
    
        def forward(self, x, maskeroo):
            x, maskeroo = self.conv1(x, maskeroo)
            x = self.nonlinear1(x)
            x, maskeroo = self.conv2(x, maskeroo)
            x = self.nonlinear2(x)
            return x, maskeroo
    
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TwoCon(1, 48)
            self.conv2 = TwoCon(48, 48)
            self.conv3 = TwoCon(48, 48)
            self.conv4 = TwoCon(48, 48)  
            self.conv5 = TwoCon(48, 48) 
            self.conv7 = TwoCon(48, 48) 
            self.conv6 = nn.Conv2d(48,1,1)
            
    
        def forward(self, x, maskero):
            x, maskero = self.conv1(x,maskero)
            x, maskero = self.conv2(x,maskero)
            x, maskero = self.conv3(x, maskero)
            x, maskero = self.conv4(x, maskero)
            x, maskero = self.conv5(x, maskero)
            x, maskero = self.conv7(x, maskero)
            x = torch.sigmoid(self.conv6(x))
            return x
        
    file_list = [f for f in os.listdir(folder)]
    
    for v in range(len(file_list)):
        
        file_name =  file_list[v]
        if file_name[0] == '.':
            continue
        start_time = time.time()
        if v>0:
            print("--- %s seconds ---" % (time.time() - start_time))
            print('100% complete')
        print(file_name)

        
        
        inp = imread(folder + '/' + file_name)
        if inp.shape[-1] == 3:
            inp = np.swapaxes(inp, -2, -1)
            inp = np.swapaxes(inp, -3, -2)
        ogshape = inp.shape

        inp = inp.reshape(-1,ogshape[-2],ogshape[-1])
        out = np.zeros(inp.shape, dtype=np.float32)
        for oz in range(inp.shape[0]):
            start_time = time.time()
            
            if oz>0:
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")

            
            print('Slice '+str(oz+1)+'/'+str(inp.shape[0]))
            print('0% complete (estimated)')
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")

            img = inp[oz,:,:]
            typer = type(inp[0,0,0])
        
            minner = np.amin(img)
            img = img - minner
            maxer = np.amax(img)
            img = img/maxer
            img = img.astype(np.float32)
            shape = img.shape
            
            
            
            Zshape = [shape[0],shape[1]]
            if shape[0] % 2 == 1:
                Zshape[0] -= 1
            if shape[1] % 2 == 1:
                Zshape[1] -= 1  
            imgZ = img[:Zshape[0],:Zshape[1]]
            
            imgZ = imgZ[:,:]
            shape = imgZ.shape
            
            whitet = np.zeros(imgZ.shape)
            blackt = np.zeros(imgZ.shape)
            
    
            
            for i in range(whitet.shape[0]):
                for j in range(whitet.shape[1]):                
                    if (i + j) % 2 == 0:
                        blackt[i,j] = 1
                    else:
                        whitet[i,j] = 1
            
            black = torch.from_numpy(blackt).to(device)
            white = torch.from_numpy(whitet).to(device)
                        
                        
            white_crop = white[2:-2,2:-2]
            black_crop = black[2:-2,2:-2]
            
            white = white.view(-1)
            black = black.view(-1)
            
            white_crop = white_crop.reshape(-1)
            black_crop = black_crop.reshape(-1)
            
            white = torch.where(white==1)
            black = torch.where(black==1)
            
            white_crop = torch.where(white_crop==1)
            black_crop = torch.where(black_crop==1)
            
            unfold5 = nn.Unfold(kernel_size=(5, 5))
            
            
            
            paddedimg = torch.from_numpy(imgZ).to(device)
            paddedimg = torch.unsqueeze(paddedimg,0)
            paddedimg = torch.unsqueeze(paddedimg,0)
            
            
            
            counterimg = torch.zeros(imgZ.shape,dtype=torch.float).to(device)
            counterimg = counterimg.view(-1)
            
            increments = torch.arange(counterimg.shape[0]//2).float().to(device)
            
            counterimg[black] = increments
            counterimg[white] = increments
            
            indixes = counterimg.view(1,1,imgZ.shape[0],imgZ.shape[1])
            
            paddedimg = F.pad(paddedimg, (2,2,2,2), "constant",-1)
            indixes = F.pad(indixes, (2,2,2,2), "constant", -1)
            
            paddedimg = unfold5(paddedimg).view(5,5,-1)
            indixes = unfold5(indixes).view(5,5,-1)
            
            left = (paddedimg[3,2]-paddedimg[3,1])**2 + (paddedimg[1,2]-paddedimg[1,1])**2
            right = (paddedimg[3,2]-paddedimg[3,3])**2 + (paddedimg[1,2]-paddedimg[1,3])**2
            down = (paddedimg[2,3]-paddedimg[3,3])**2 + (paddedimg[2,1]-paddedimg[3,1])**2
            up = (paddedimg[2,3]-paddedimg[1,3])**2 + (paddedimg[2,1]-paddedimg[1,1])**2
            
            
            #^Problem b/c of possible negative?
            
            
            leftind = indixes[2,1]
            rightind = indixes[2,3]
            downind = indixes[3,2]
            upind = indixes[1,2]
            
            inindixes = indixes[2,2]
            
            left_black = torch.clone(left[black]) 
            right_black = torch.clone(right[black])
            up_black = torch.clone(up[black])
            down_black = torch.clone(down[black])
            
            leftind_black = torch.clone(leftind[black]) 
            rightind_black = torch.clone(rightind[black])
            upind_black = torch.clone(upind[black])
            downind_black = torch.clone(downind[black])
            
            inindixes_black = torch.cat([torch.clone(inindixes[black]),torch.clone(inindixes[black]),torch.clone(inindixes[black]),torch.clone(inindixes[black])])
            paddedimg_black = torch.cat([left_black,right_black,down_black,up_black])
            outindixes_black = torch.cat([leftind_black,rightind_black,downind_black,upind_black])
            
            filt_black = torch.where(outindixes_black>-1)
            
            inindixes_black = torch.clone(inindixes_black[filt_black])
            outindixes_black = torch.clone(outindixes_black[filt_black]) 
            paddedimg_black = torch.clone(paddedimg_black[filt_black]) 
            
            min_black = torch.min(paddedimg_black)
            paddedimg_black = paddedimg_black-min_black
            max_black = torch.max(paddedimg_black)
            paddedimg_black = paddedimg_black/max_black
            paddedimg_black = torch.clamp(paddedimg_black,0,1)
            paddedimg_black = paddedimg_black*99+1
            paddedimg_black = paddedimg_black.long()
            
            rows_black = inindixes_black.cpu().detach().numpy()
            values_black = paddedimg_black.cpu().detach().numpy()
            cols_black = outindixes_black.cpu().detach().numpy()
            
            coo_black = coo_matrix((values_black, (rows_black, cols_black)), shape=(imgZ.shape[0]*imgZ.shape[1]//2,imgZ.shape[0]*imgZ.shape[1]//2))
            
            match_black = ssc.min_weight_full_bipartite_matching(coo_black)
            
            imgflat_black = torch.from_numpy(imgZ.copy()).to(device).view(-1)
            imgflat_black[black] = imgflat_black[white][match_black[1]]
    
            left_white = left[white] 
            right_white = right[white]
            up_white = up[white]
            down_white = down[white]
            
            leftind_white = leftind[white] 
            rightind_white = rightind[white]
            upind_white = upind[white]
            downind_white = downind[white]
            
            inindixes_white = torch.cat([inindixes[white],inindixes[white],inindixes[white],inindixes[white]])
            paddedimg_white = torch.cat([left_white,right_white,down_white,up_white])
            outindixes_white = torch.cat([leftind_white,rightind_white,downind_white,upind_white])  
            
            filt_white= torch.where(outindixes_white>-1)
            
            inindixes_white = torch.clone(inindixes_white[filt_white])
            outindixes_white = torch.clone(outindixes_white[filt_white]) 
            paddedimg_white = torch.clone(paddedimg_white[filt_white]) 
            
            min_white = torch.min(paddedimg_white)
            paddedimg_white = paddedimg_white-min_white
            max_white = torch.max(paddedimg_white)
            paddedimg_white = paddedimg_white/max_white
            paddedimg_white = torch.clamp(paddedimg_white,0,1)
            paddedimg_white = paddedimg_white*99+1
            paddedimg_white = paddedimg_white.long()
            
            rows_white = inindixes_white.cpu().detach().numpy()
            values_white = paddedimg_white.cpu().detach().numpy()
            cols_white = outindixes_white.cpu().detach().numpy()
            
            coo_white = coo_matrix((values_white, (rows_white, cols_white)), shape=(imgZ.shape[0]*imgZ.shape[1]//2,imgZ.shape[0]*imgZ.shape[1]//2))
            
            match_white = ssc.min_weight_full_bipartite_matching(coo_white)
            
            imgflat_white = torch.from_numpy(imgZ.copy()).to(device).view(-1)
            imgflat_white[white] = imgflat_white[black][match_white[1]]
      
            
            
    
            
            augment = []
            for i in range(len(functions)):
                khufu = functions[i](img.copy())
                khufu = torch.from_numpy(khufu.copy())
                khufu = torch.unsqueeze(khufu,0)
                khufu = torch.unsqueeze(khufu,0).to(device)
                augment.append(khufu)
    
      
        
            img = torch.from_numpy(img)
            img = torch.unsqueeze(img,0)
            img = torch.unsqueeze(img,0)
            img = img.to(device)
    
            #listimg = listimgV1+listimgV2+listimgV3+listimgV4
            
            net = Net()
            net.to(device)
            criterion = nn.BCELoss(reduction='none')
            optimizer = optim.Adam(net.parameters(), lr=0.0005)
    
            bothap = imgflat_black.view(1,1,imgZ.shape[0],imgZ.shape[1])
            imgup =  imgflat_white.view(1,1,imgZ.shape[0],imgZ.shape[1])
            bothapcpu = (bothap[0,0,:,:].cpu().detach().numpy()*maxer+minner)*blackt
            imgupcpu =  (imgup[0,0,:,:].cpu().detach().numpy()*maxer+minner)*whitet
            running_loss1=0.0
            last10mask = []
            last10blind = []
            for sp in range(len(ifunctions)):
                last10blind.append(functions[sp](np.zeros_like(img[0,0,:,:].cpu().detach().numpy())))
                last10mask.append(functions[sp](np.zeros_like(img[0,0,:,:].cpu().detach().numpy())))
            keepall = [np.zeros_like(bothap[0,0,:,:].cpu().detach().numpy()),
                       np.zeros_like(bothap[0,0,:,:].cpu().detach().numpy())]
    
            keepallcounter = 0
            keepallcounter2 = 0
            cleaned = 0
            counter = 0
            walk = 0
            lastpct = None
            goodo = True
            lastval = None
            maxwalk = -1
            newval = np.inf
            countdown = 1
            
            countso = 500
    
            while goodo:
                counter += 1
                switcher = np.random.randint(len(functions))
                
                tranimg = augment[switcher]
                
                masker = torch.rand(tranimg.shape)
                masker = torch.where(masker>0.8)
                maskerto = torch.ones_like(tranimg).to(device)
                maskerto[masker] = 0
                
                leaker = torch.rand(tranimg.shape)
                leaker = torch.where(leaker>0.999)
                leakerto = torch.zeros_like(tranimg).to(device)
                leakerto[leaker] = 1
                
                inputs = tranimg
                inputs = tranimg*maskerto
                labello = tranimg
                
                optimizer.zero_grad()
                outputs = net(inputs,maskerto)
                
                loss1 = torch.mean(((1.0-maskerto)+leakerto)*criterion(outputs, labello))
                
                
                if counter > countso:
                    
                    with torch.no_grad():
    
                        tempoutputs = outputs*maxer+minner
                        lasts2s =  torch.zeros_like(outputs).to(device)
                        lasts2s[torch.where(maskerto==1)] = tempoutputs[torch.where(maskerto==1)]
                        lasts2s = lasts2s[0,0,:,:].cpu().detach().numpy()
                        lastmasker = maskerto[0,0,:,:].cpu().detach().numpy()
                    
                    
                    last10blind[switcher] += lasts2s
                    last10mask[switcher] += lastmasker
                    
                    
    
                loss = loss1
                running_loss1+=loss1.item()
                loss.backward()
                optimizer.step()
                
                
                running_loss1=0.0
    
                if counter > countso:
                
                    with torch.no_grad():
                        
                        
                        coinflip = counter % 2
                        if coinflip == 0:
                            imgdown = bothap
                            
                            maskert0 = torch.ones_like(imgdown)
                            outputs = net(imgdown,maskert0)
                            outputs = outputs[0,0,:,:].cpu().detach().numpy()
                            outputs = outputs*maxer+minner
                            keepall[0] += outputs
                            keepallcounter+=1
                        elif coinflip == 1:
                            imgdown = imgup
                            
                            maskert0 = torch.ones_like(imgdown)
                            outputs = net(imgdown,maskert0)
                            outputs = outputs[0,0,:,:].cpu().detach().numpy()
                            outputs = outputs*maxer+minner
                            keepall[1] += outputs
                            keepallcounter2+=1
                            
    
                if counter == 750:
                    print('6% complete (estimated)')
                    sys.stdout.write("\033[F")
                    sys.stdout.write("\033[K")
                if counter % countso == (countso-1) and counter > countso:
                    
                    
                    H0 = keepall[0]/keepallcounter
                    V0 = keepall[1]/keepallcounter2
                    H0 = H0*whitet
                    V0 = V0*blackt
                    unfold64 = nn.Unfold(kernel_size=(32, 32), stride=32)
                    H0 = torch.from_numpy(H0).view(1,1,imgZ.shape[0],imgZ.shape[1]).to(device)
                    V0 = torch.from_numpy(V0).view(1,1,imgZ.shape[0],imgZ.shape[1]).to(device)
                    
                    H0 = unfold64(H0)
                    V0 = unfold64(V0)
                    
                    imgupcpunow = torch.from_numpy(imgupcpu).view(1,1,imgZ.shape[0],imgZ.shape[1]).to(device)
                    bothapcpunow = torch.from_numpy(bothapcpu).view(1,1,imgZ.shape[0],imgZ.shape[1]).to(device)
                    
                    imgupcpunow = unfold64(imgupcpunow)
                    bothapcpunow = unfold64(bothapcpunow)
                    
                    newval1 = torch.sum((imgupcpunow - H0)**2,axis=1)[0,:].cpu().detach().numpy()
                    newval2 = torch.sum((bothapcpunow - V0)**2,axis=1)[0,:].cpu().detach().numpy()
                    
                    
                    newval = newval1+newval2
                    
    
                    if lastval is None:
                        lastval = newval
                    curpct = (np.sum(newval>lastval)/len(newval>lastval))
                    if lastpct is None:
                        lastpct = curpct
                    if curpct == 0:
                        walk+=0
                    elif curpct == 1:
                        goodo=False
                    elif curpct >= lastpct:
                        walk +=1
                        if maxwalk < walk:
                            maxwalk = walk
                            print(str(int(100*(maxwalk+1)/16))+'% complete (estimated)')
                            sys.stdout.write("\033[F")
                            sys.stdout.write("\033[K")
                            
                    else:
                        walk = max(0,walk-1)
                    lastpct = curpct
                    if walk >= 15:
                        goodo = False
    
                    
                    if  goodo == False:
                        for iw in range(len(ifunctions)):
                            last10blind[iw] = ifunctions[iw](last10blind[iw])
                            last10mask[iw] = ifunctions[iw](last10mask[iw])
                        last10blind = np.sum(np.stack(last10blind), axis=0)
                        last10mask = np.sum(np.stack(last10mask), axis=0)
                        H2 = last10blind/last10mask
                        out[oz,:,:] = H2
                        
            torch.cuda.empty_cache()
                    
        out = out.reshape(ogshape)
        imwrite(outfolder + '/' + file_name, out.astype(typer), imagej=True)
            
            
    
    
            
            

