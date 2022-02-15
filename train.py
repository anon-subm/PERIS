import time
import torch
import numpy as np
import argparse
import random
from models import PERIS
from metric import cal_measures
from dataloaders.dataloader import DataLoader

torch.set_num_threads(1)

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

class Instructor:
    def __init__(self, opt):
        self.opt = opt               

        self.data_loader = DataLoader(self.opt) 

        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()
        
        opt.numuser = self.trn_loader.dataset.numuser
        opt.numitem = self.trn_loader.dataset.numitem
        self.model = self.opt.model_class(self.opt).cuda()
        
        self._print_args()
        
    def train(self):        
                
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.learning_rate, weight_decay=opt.l2reg)
        
        best_score = -1 
        best_topHits, best_topNdcgs = None, None
        batch_loss = 0
        batch_loss_aux = 0
        c = 0 # to check early stopping
        
        for epoch in range(self.opt.num_epoch):
            st = time.time()
    
            for i, batch_data in enumerate(self.trn_loader):
                batch_data = [bd.cuda() for bd in batch_data]                                 
                optimizer.zero_grad() 

                if epoch < self.opt.warmup_epochs:
                    loss = self.model.compute_warmup_loss(batch_data)          
                else:
                    loss, loss_IS = self.model.compute_loss(batch_data)                

                loss.backward()
                
                optimizer.step()
    
                batch_loss += loss.data.item()
                
                if epoch>=self.opt.warmup_epochs:
                    batch_loss_aux += loss_IS.data.item()

            elapsed = time.time() - st
            evalt = time.time()
            
            with torch.no_grad():
                topHits, topNdcgs  = cal_measures(self.vld_loader, self.model, opt, 'vld')
                
                if (topHits[10] + topNdcgs[10])/2 > best_score:
                    best_score = (topHits[10] + topNdcgs[10])/2
                    
                    best_topHits = topHits
                    best_topNdcgs = topNdcgs
                    
                    c = 0
                    
                    test_topHits, test_topNdcgs = cal_measures(
                                    self.tst_loader, self.model, opt, 'tst')                    

                evalt = time.time() - evalt 
            
            print(('(%.1fs, %.1fs)\tEpoch [%d/%d], TRN_ERR : %.4f, TRN_IS_ERR : %.4f, v_score : %5.4f, tHR@10 : %5.4f'% (elapsed, evalt, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), batch_loss_aux/len(self.trn_loader), (topHits[10] + topNdcgs[10])/2,  test_topHits[10])))

            batch_loss = 0
            batch_loss_aux = 0

            c += 1

            if epoch < self.opt.warmup_epochs:
                c = 0 # don't count patient during warm-up steps            
            
            if c > 5: break # Early-stopping
        
        print(('\nValid score@10 : %5.4f, HR@10 : %5.4f, NDCG@10 : %5.4f\n'% (((best_topHits[10] + best_topNdcgs[10])/2), best_topHits[10],  best_topNdcgs[10])))
        
        return test_topHits,  test_topNdcgs
            
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('\nn_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        print('')

    def run(self, repeats):
        results = []
        rndseed = [19427, 78036, 37498, 87299, 60330] # randomly-generated seeds
        for i in range(repeats):
            print('\n💫 run: {}/{}'.format(i+1, repeats))
            
            if self.opt.model_name in ['transperis', 'linearperis', 'peris', 'fir']:
                print('\nWarmup up to {}-th epoch\n'.format(self.opt.warmup_epochs))
            
            random.seed(rndseed[i]); np.random.seed(rndseed[i]); torch.manual_seed(rndseed[i])
            self._reset_params()
            
            results.append(ins.train())
        
        results = np.array(results)
        
        hrs_mean = np.array([list(i.values()) for i in results[:,0]]).mean(0)
        ndcg_mean = np.array([list(i.values()) for i in results[:,1]]).mean(0)
        
        hrs_std = np.array([list(i.values()) for i in results[:,0]]).mean(0)
        ndcg_std = np.array([list(i.values()) for i in results[:,1]]).mean(0)        
    
        print('*TST Performance\tTop2\tTop5\t\tTop10\t\tTop20\t')
        print('*HR means: {}'.format(', '.join(hrs_mean.astype(str))))
        print('*NDCG means: {}'.format(', '.join(ndcg_mean.astype(str))))
        
    def _reset_params(self):
        self.model = self.opt.model_class(self.opt).cuda()
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='peris', type=str)
    parser.add_argument('--dataset', default='cell', type=str)    
    parser.add_argument('--num_run', default=5, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)    
    parser.add_argument('--batch_size', default=128, type=int)    
    parser.add_argument('--l2reg', default=0.0, type=float)    
    
    parser.add_argument('--margin', default=0.6, type=float)    
    parser.add_argument('--K', default=50, type=int)      
    parser.add_argument('--numneg', default=5, type=int)  
    
    parser.add_argument('--lamb', default=0.2, type=float)
    parser.add_argument('--mu', default=0.2, type=float)
   
    parser.add_argument('--binsize', default=8, type=int)
    parser.add_argument('--period', default=64, type=int)      
    parser.add_argument('--tau', default=0, type=float)
    parser.add_argument('--bin_ratio', default=0.5, type=float)    
    parser.add_argument('--neg_weight', default=1.0, type=float)        
    parser.add_argument('--warmup_epochs', default=5, type=int)    
    parser.add_argument('--maxhist', default=100, type=int)
    
    opt = parser.parse_args()
    
    torch.cuda.set_device(opt.gpu)
    
    model_classes = { 
        'peris':PERIS, 
    }  
    
    dataset_path = './data/{}/rec'.format(opt.dataset)
    
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_path = dataset_path

    ins = Instructor(opt)
    
    ins.run(opt.num_run)     
