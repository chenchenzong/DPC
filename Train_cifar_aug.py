from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar_aug as dataloader

from loss import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
parser.add_argument('--id', default='cifar10')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./dataset/cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def conv_p(logits):
    # 10/n_class
    alpha_t = torch.exp(logits)+10./args.num_class
    total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)
    expected_p = alpha_t / total_alpha_t
    return expected_p

def consistency_loss(output1, output2):            
    preds1 = conv_p(output1).detach()
    preds2 = torch.log(conv_p(output2))
    loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
    loss_kldiv = torch.sum(loss_kldiv, dim=1)
    return loss_kldiv

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    #if len(labeled_trainloader.dataset) >= len(unlabeled_trainloader.dataset):
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)
    #else:
    #    num_iter = (len(unlabeled_trainloader.dataset)//args.batch_size)

    for batch_idx in range(num_iter):
        try:
            inputs_x, inputs_x2, inputs_x_aug, inputs_x2_aug, labels_x, w_x  = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader) 
            inputs_x, inputs_x2, inputs_x_aug, inputs_x2_aug, labels_x, w_x  = labeled_train_iter.next()
        try:
            inputs_u, inputs_u2, inputs_u_aug, inputs_u2_aug, w_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u_aug, inputs_u2_aug, w_u = unlabeled_train_iter.next()                
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 
        w_u = w_u.view(-1,1).type(torch.FloatTensor)

        inputs_x, inputs_x2, inputs_x_aug, inputs_x2_aug, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x_aug.cuda(), inputs_x2_aug.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u_aug, inputs_u2_aug, w_u = inputs_u.cuda(), inputs_u2.cuda(), inputs_u_aug.cuda(), inputs_u2_aug.cuda(), w_u.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11,_ = net(inputs_u)
            outputs_u12,_ = net(inputs_u2)
            outputs_u21,_ = net2(inputs_u)
            outputs_u22,_ = net2(inputs_u2)          
            
            pu = (conv_p(outputs_u11) + conv_p(outputs_u12) + conv_p(outputs_u21) + conv_p(outputs_u22)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()    
            
            # label refinement of labeled samples
            outputs_x,_ = net(inputs_x)
            outputs_x2,_ = net(inputs_x2)
            
            px = (conv_p(outputs_x) + conv_p(outputs_x2)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_inputs_aug = torch.cat([inputs_x_aug, inputs_x2_aug, inputs_u_aug, inputs_u2_aug], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        input_a_aug, input_b_aug = all_inputs_aug, all_inputs_aug[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b       
        mixed_input_aug = l * input_a_aug + (1 - l) * input_b_aug   
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits,logits2 = net(mixed_input)
        logits_aug, logits2_aug, _ = net(mixed_input_aug)
        logits_x = logits[:batch_size*2]
        logits_u = logits2[batch_size*2:]
        logits_x_aug = logits_aug[:batch_size*2]
        logits_u_aug = logits2_aug[batch_size*2:]
           
        Lx, Lu, lamb = criterion(logits_x, l, target_a[:batch_size*2].argmax(1), target_b[:batch_size*2].argmax(1), logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
    
        num_logits = logits.shape[0]
        if args.dataset == "cifar10":
            Con = 3 * torch.mean(consistency_loss(logits, logits_aug))
        else:
            Con = 1 * torch.mean(consistency_loss(logits, logits_aug))

        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = conv_p(logits).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty + Con
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs,_ = net(inputs)         
        loss1, loss2 = EDL_Loss(args.num_class)(outputs,labels)
        loss = loss1.mean()+loss2.mean()

        if args.noise_mode=='asym':
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1,_ = net1(inputs)
            outputs2,_ = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def eval_train(model):    
    model.eval()
    margins = torch.zeros(50000) 
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs,_ = model(inputs) 

            for b in range(inputs.size(0)):
                evidence_pos = outputs[b,targets[b]]
                copy_outputs = outputs[b].clone()
                copy_outputs[targets[b]] = -1e5
                evidence_neg = copy_outputs.max()
                margins[index[b]]=evidence_pos-evidence_neg
    
    margins = (margins-margins.min())/(margins.max()-margins.min())

    input_margin = margins.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_margin) 
    prob1 = prob[:,gmm.means_.argmax()]   

    return prob1

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, l, targets_x1, targets_x2, outputs_u, targets_u, epoch, warm_up):
        probs_u = conv_p(outputs_u)

        Lx11, Lx12 = EDL_Loss(args.num_class)(outputs_x,targets_x1)
        Lx21, Lx22 = EDL_Loss(args.num_class)(outputs_x,targets_x2)

        Lx = (l*Lx11 + (1-l)*Lx21).mean() + (l*Lx12 + (1-l)*Lx22).mean()
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = conv_p(outputs)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

stats_log=open('./checkpoint/%s_%.1f_%s_%s'%(args.dataset,args.r,args.noise_mode,str(args.seed))+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.1f_%s_%s'%(args.dataset,args.r,args.noise_mode,str(args.seed))+'_acc.txt','w')     

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:         
        prob1=eval_train(net1)    
        prob2=eval_train(net2)         
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)    
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2         

    test(epoch,net1,net2)  


