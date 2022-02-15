import csv
import time
import torch
import random
import numpy as np

from tqdm import tqdm
from torch.utils import data
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from dateutil.relativedelta import relativedelta


random.seed(2022)

def toymd(time):
    return datetime.utcfromtimestamp(time)#.strftime('%Y-%m-%d')

def get_timebin_from_timestamp(times, timebins): # times: timestamps
    binfeature = np.zeros(len(timebins))
    for eachtime in times:        
        binidxs = np.where(timebins <= toymd(eachtime))[0] # select an element
        
        if len(binidxs) != 0:          
            target_bin_idx = binidxs[-1] 
            binfeature[target_bin_idx] += 1    
    return binfeature

def get_timebin_from_timestamp4eachuser(times, timebins): # times: timestamps of consumed items        
    time_bins = [] # frequency bins that will be used as features
    for eachtime in times:   
        binfeature = np.zeros(len(timebins))

        binidxs = np.where(timebins <= toymd(eachtime))[0] # select an element
        
        if len(binidxs) != 0:          
            target_bin_idx = binidxs[-1] 
            binfeature[target_bin_idx] += 1   
        time_bins.append(binfeature)
    return time_bins

class SEQRS_Dataset(data.Dataset):
    
    def build_consumption_history(self, uirt):
        # Build a dictionary for user: items consumed by the user
        uidict = {}
        allitems = set()
        for u, i, _, _ in uirt.astype(int):
            if u not in uidict: uidict[u] = set()
            uidict[u].add(i)
            allitems.add(i)
            
        ui_cand_dict = {}    
        for u in tqdm(uidict):
            allnegitems = list(allitems - uidict[u])
            truncated_negitems = allnegitems[:1000]
            ui_cand_dict[u] = np.array(truncated_negitems)
#             ui_cand_dict[u] = np.array(list(allitems - uidict[u]))
            
            # TODO: Its size increases proportionally 
        
        return uidict, allitems, ui_cand_dict
        
    def __init__(self, path, trn_numneg, opt): 
        dpath = '/'.join(path.split('/')[:-1])
        if dpath[-1] != '/': dpath += '/'
        dtype = path.split('/')[-1] # dtype {trn, vld, tst}
        
        self.opt = opt
        
        st = time.time()                        
        
        if dtype == 'trn': 
            self.numneg = trn_numneg            
            
        self.uirt = np.load(dpath+dtype)
        
        # userhist contains both items and times
        if dtype != 'tst': # trainig or validation
            userhist = np.load(dpath+'userhist', allow_pickle=True).item()                
            itemhist = np.load(dpath+'itemhist', allow_pickle=True).item()
            neighborhist = np.load(dpath+'neighborhist', allow_pickle=True).item()

            itlist = [[iid, itemhist[iid]] for iid in itemhist]
                
        elif dtype == 'tst':
            userhist = np.load(dpath+'userhist_wvld', allow_pickle=True).item()          
            itemhist = np.load(dpath+'itemhist_wvld', allow_pickle=True).item()      
            neighborhist = np.load(dpath+'neighborhist_wvld', allow_pickle=True).item()    
            
            itlist = [[iid, itemhist[iid]] for iid in itemhist]
            
        # recent item first
        for u in userhist: # + truncation wiht MAX # of users' hist
            itemseq, timeseq = userhist[u]
            reverse_idx = (-timeseq).argsort()            
            userhist[u] = [itemseq[reverse_idx][:opt.maxhist], timeseq[reverse_idx][:opt.maxhist]] # NOTE: trauncation from the front is right because the descending sorting           
        for i in neighborhist: # + truncation wiht MAX # of users' hist
            userseq, timeseq = neighborhist[i]            
            reverse_idx = (-timeseq).argsort()
            neighborhist[i] = [userseq[reverse_idx][:opt.maxhist], timeseq[reverse_idx][:opt.maxhist]]                        
        for i in itemhist:
            timeseq = itemhist[i]
            reverse_idx = (-timeseq).argsort()            
            itemhist[i] = timeseq[reverse_idx][:opt.maxhist]   
            # itemhist[i] = timeseq[reverse_idx] # NOTE This version works well..

        # 3.1 Building bin basis
        trntimes = np.load(opt.dataset_path + '/trn')[:,-1].astype(int) # ascending sorted
        mintime = toymd(min(trntimes))
        
        timedelta = relativedelta(weeks=opt.binsize)

        if dtype == 'trn':
            trnfront_time = trntimes.max() - 60 * 60 * 24 * 7 * opt.period 
            trnfront_idx = np.where(trntimes < trnfront_time)[0][-1]        
            self.label_period_start_time = trntimes[trnfront_idx]
            trnlabeltime = toymd(self.label_period_start_time)
            
            bins = np.array([mintime + timedelta*i for i in range(1000) # quick implementation
                         if mintime + timedelta*i < trnlabeltime] + [trnlabeltime])  
            
            self.bins = bins[-int(opt.bin_ratio * len(bins)):] # Time bin truncation
        elif dtype == 'vld':
            trnmaxtime = toymd(max(trntimes))
            
            bins = np.array([mintime + timedelta*i for i in range(1000) # quick implementation
                         if mintime + timedelta*i < trnmaxtime] + [trnmaxtime])  
                
            self.bins = bins[-int(opt.bin_ratio * len(bins)):]   
        elif dtype == 'tst':
            vld = [l for l in csv.reader(
                open('/'.join(opt.dataset_path.split('/')[:-1])+'/split/vld.csv'))]
            vldtimes = np.array(vld)[:,-1].astype(int)                       


            vldmaxtime = toymd(max(vldtimes))        

            bins = np.array([mintime + timedelta*i for i in range(1000) # quick implementation
                         if mintime + timedelta*i < vldmaxtime] + [vldmaxtime])   
            self.bins = bins[-int(opt.bin_ratio * len(bins)):]


        # 3.2 Build time bins for each user's all consumed items        
        self.freqbins = {}
        for u in userhist:            
            consumed_iids = userhist[u][1]
            fb = get_timebin_from_timestamp4eachuser(consumed_iids, self.bins)            
            self.freqbins[u] = np.array(fb)      

        self.neighbor_user_freqbins = {}
        for i in neighborhist:        
            self.neighbor_user_freqbins[i] = {}    
            consumed_users, consumed_times = neighborhist[i]
            fb = get_timebin_from_timestamp4eachuser(consumed_times, self.bins)  
            
            self.neighbor_user_freqbins[i]['user'] = consumed_users
            self.neighbor_user_freqbins[i]['time'] = consumed_times
            self.neighbor_user_freqbins[i]['freqbin'] = np.array(fb)
            

        # Data building for training/validation/test    
        if dtype == 'trn':                      
            trndata = [[u, userhist[u]] for u in userhist]  
            
            
            # #.Time bin processing for items
            trntimes = np.concatenate([i[1] for i in itlist]).astype(int)
            
            mintime = toymd(min(trntimes))
            trnmaxtime = toymd(max(trntimes))

            trnfront_time = trntimes.max() - 60 * 60 * 24 * 7 * opt.period 
            trnfront_idx = np.where(trntimes < trnfront_time)[0][-1]        
            label_period_st = trntimes[trnfront_idx] # st: start time
          

            self.item_fldict = {} # dictionary of (Item ID - feature, label)
            # item_feature, item_label = [], [] 
            for iid in itemhist:
                item_time = itemhist[iid]

                feature = item_time[item_time<label_period_st]  
                
                label = int(bool(sum(item_time >= label_period_st)))

                binfeature = get_timebin_from_timestamp(feature, self.bins)

                self.item_fldict[iid] = [binfeature, label]


            self.first, self.second = zip(*trndata) # user and their sequential items with times
            self.third = np.zeros(len(self.first)) # Negative items. 'train_collate' does the job. 
            
            # # Global maximum sacrifies some computation
            maxnum_neighbor = max([len(self.neighbor_user_freqbins[i]['user']) for i in self.neighbor_user_freqbins])
            maxnum_neighbor_time = max([len(self.neighbor_user_freqbins[i]['time']) for i in self.neighbor_user_freqbins])
            maxnum_neighbor_freqbin = max([len(self.neighbor_user_freqbins[i]['freqbin']) for i in self.neighbor_user_freqbins])

            self.fourth = []
            for i in range(len(self.first)):
                u = self.first[i]
                all_neighbors = []
                all_neighbors_time = []
                all_neighbors_freqbin =[]

                for j in self.second[i][0]:
                    neighbors = torch.LongTensor(self.neighbor_user_freqbins[j]['user'])
                    neighbors_time = torch.LongTensor(self.neighbor_user_freqbins[j]['time'])
                    neighbors_freqbin = torch.LongTensor(self.neighbor_user_freqbins[j]['freqbin'])

                    all_neighbors.append(neighbors)
                    all_neighbors_time.append(neighbors_time)
                    all_neighbors_freqbin.append(neighbors_freqbin)
                

                # First make sure padding to the maximum length
                all_neighbors[0] = torch.cat([all_neighbors[0], torch.zeros(maxnum_neighbor-len(all_neighbors[0]))])
                all_neighbors_time[0] = torch.cat([all_neighbors_time[0], torch.zeros(maxnum_neighbor_time-len(all_neighbors_time[0]))])
                all_neighbors_freqbin[0] = torch.cat([all_neighbors_freqbin[0], torch.zeros(maxnum_neighbor_freqbin-all_neighbors_freqbin[0].shape[0],all_neighbors_freqbin[0].shape[1])])

                all_neighbors = pad_sequence(all_neighbors, batch_first=True)
                all_neighbors_time = pad_sequence(all_neighbors_time, batch_first=True)
                all_neighbors_freqbin = pad_sequence(all_neighbors_freqbin, batch_first=True)

                self.fourth.append({'user':all_neighbors, 'time':all_neighbors_time, 'freqbin':all_neighbors_freqbin})
            
            self.numuser = len(set(self.uirt[:,0]))
            self.numitem = len(set(self.uirt[:,1]))
                                            

        elif dtype in ['vld', 'tst']: # Evaluation
            # Find user histories 
            
            # Build validation data for ranking evaluation
            newuir = []
            for row in self.uirt:                                
                user = row[0]
                uhist = userhist[user]                
                items = row[1:] # 1 positive item + 100 negative items
                newuir.append([[user, uhist], items])
                
            self.first, self.second = zip(*newuir)
            self.third = np.zeros(len(self.first)) # this is dummy data
            self.fourth = np.zeros(len(self.first)) # this is dummy data

             # # Global maximum sacrifies some computation
            maxnum_neighbor = max([len(self.neighbor_user_freqbins[i]['user']) for i in self.neighbor_user_freqbins])
            maxnum_neighbor_time = max([len(self.neighbor_user_freqbins[i]['time']) for i in self.neighbor_user_freqbins])
            maxnum_neighbor_freqbin = max([len(self.neighbor_user_freqbins[i]['freqbin']) for i in self.neighbor_user_freqbins])

            users = [i[0] for i in self.first]
            items = [row[1:] for row in self.uirt]
            self.fourth = []
            for i in range(len(users)):
                u = users[i]
                all_neighbors = []
                all_neighbors_time = []
                all_neighbors_freqbin =[]

                for j in items[i]:
                    neighbors = torch.LongTensor(self.neighbor_user_freqbins[j]['user'])
                    neighbors_time = torch.LongTensor(self.neighbor_user_freqbins[j]['time'])
                    neighbors_freqbin = torch.LongTensor(self.neighbor_user_freqbins[j]['freqbin'])

                    all_neighbors.append(neighbors)
                    all_neighbors_time.append(neighbors_time)
                    all_neighbors_freqbin.append(neighbors_freqbin)

                # Pdding here                

                # First make sure padding to the maximum length
                all_neighbors[0] = torch.cat([all_neighbors[0], torch.zeros(maxnum_neighbor-len(all_neighbors[0]))])
                all_neighbors_time[0] = torch.cat([all_neighbors_time[0], torch.zeros(maxnum_neighbor_time-len(all_neighbors_time[0]))])
                all_neighbors_freqbin[0] = torch.cat([all_neighbors_freqbin[0], torch.zeros(maxnum_neighbor_freqbin-all_neighbors_freqbin[0].shape[0],all_neighbors_freqbin[0].shape[1])])

                all_neighbors = pad_sequence(all_neighbors, batch_first=True)
                all_neighbors_time = pad_sequence(all_neighbors_time, batch_first=True)
                all_neighbors_freqbin = pad_sequence(all_neighbors_freqbin, batch_first=True)

                self.fourth.append({'user':all_neighbors, 'time':all_neighbors_time, 'freqbin':all_neighbors_freqbin})


            self.item_fdict = {} # dictionary of (Item ID - feature)            
            for iid in itemhist:    
                item_time = itemhist[iid]            
                feature = item_time
                
                # make time stamps to timebins
                binfeature = get_timebin_from_timestamp(feature, self.bins)
                
                self.item_fdict[iid] = binfeature

            
        
        print('{} data : {:.2}s'.format(dtype.upper(), time.time()-st))

    def __getitem__(self, index):
        # Training: [user, positive, negative]
        # Testing: [user, canidate item, label] 
        return self.first[index], self.second[index], self.third[index], self.fourth[index]
    
    def __len__(self):
        """Returns the total number of user-item pairs."""
        return len(self.first)
  
  
    def train_collate(self, batch):
        # Input: [user, postive item, dummy]
        # Output: [user, positive item, negative item]
        batch = [i for i in filter(lambda x:x is not None, batch)]
        
        uids, items_times, _, neighbor_info = zip(*batch) # users and their history    
        iids, raw_times = zip(*items_times)    

        flatten_user = np.concatenate([[uids[i]]*len(iids[i]) for i in range(len(uids))])
        flatten_item = np.concatenate(iids)        
        flatten_time = np.concatenate(raw_times)

        flatten_history = [[iids]*len(iids[i]) for i in range(len(iids))]

        # User's consumption history (items) to compute virtual features
        num_items = torch.LongTensor([len(i) for i in iids]) 
        pad_items = pad_sequence([torch.LongTensor(it) for it in iids], batch_first=True)
        pad_items = torch.repeat_interleave(pad_items, num_items, dim=0)

        # Making data into tensor
        users = torch.LongTensor(flatten_user)
        items = torch.LongTensor(flatten_item)
        times = torch.LongTensor(flatten_time)

        
        pack_times = pad_sequence([torch.FloatTensor(ti) for ti in raw_times], batch_first=True)

        # NOTE: randomly-selecting negative samples shows almost similar performance to using items not consumed by users, while the random-sampling drastically reduces the computation time
        all_neg_items = np.random.randint(self.opt.numitem, size=(self.opt.numneg, len(users)))
        all_neg_items = torch.LongTensor(all_neg_items)

        # Flatten item (extrinsic data)       
        flatten_items = items # Not matrix format but redundant data

        # Find feature & label for extrinsic prediction
        item_freqbins, item_labels = [], []
        for iid in flatten_items.tolist():
            ft, lb = self.item_fldict[iid]
            item_freqbins.append(ft)
            item_labels.append(lb)
            
        item_freqbins = pad_sequence([torch.LongTensor(itf) for itf in item_freqbins], batch_first=True)
        item_labels = torch.LongTensor(item_labels)   
        
        # Intrinsic frequncy bins
        freqbins = [self.freqbins[u] for u in uids]        
        pad_freqbins = pad_sequence([torch.FloatTensor(fb) for fb in freqbins], batch_first=True)   
        pad_freqbins = torch.repeat_interleave(pad_freqbins, num_items, dim=0)

        pad_pack_times = torch.repeat_interleave(pack_times, num_items, dim=0)

        # data for neighborhodd (Extrinsic prediction)

        pad_neighbors = torch.cat([neighbor_info[i]['user'] for i in range(len(neighbor_info))], dim=0).long()
        pad_neighbors_time = torch.cat([neighbor_info[i]['time'] for i in range(len(neighbor_info))], dim=0)
        pad_neighbors_freqbin = torch.cat([neighbor_info[i]['freqbin'] for i in range(len(neighbor_info))], dim=0)

        return users, items, times, pad_freqbins, pad_items, pad_pack_times, all_neg_items, item_freqbins, item_labels, pad_neighbors, pad_neighbors_time, pad_neighbors_freqbin


    def test_collate(self, batch):
        batch = [i for i in filter(lambda x:x is not None, batch)]

        uinfo, cand_items, labels, neighbor_info = zip(*batch)      
        
        uid, items_times = zip(*uinfo)
        items, times = zip(*items_times)

        users = torch.LongTensor(uid)
        labels = torch.LongTensor(labels)
        cand_items = torch.LongTensor(np.array(cand_items))    
        items = pad_sequence([torch.LongTensor(it) for it in items], batch_first=True)
        times = pad_sequence([torch.LongTensor(ti.astype(int)) for ti in times], batch_first=True)
        
        flat_canditems = cand_items.flatten().tolist()

        item_freqbins = torch.FloatTensor(np.array([self.item_fdict[i] for i in flat_canditems]))        

        # Frequncy bins         
        freqbins = [self.freqbins[u] for u in uid]
        maxfb = max([fb.shape[0] for fb in freqbins])        
                
        # Padding frequency bins
        numbins = len(self.bins)
        pad_freqbins = []
        for fb in freqbins:
            zeropad = np.zeros((maxfb - fb.shape[0], numbins))
            fb = np.concatenate([fb, zeropad])
            pad_freqbins.append(fb)
        pad_freqbins = np.array(pad_freqbins)
        pad_freqbins = torch.FloatTensor(pad_freqbins)
        
        all_user_info = [users, items, times, pad_freqbins]   
        
        pad_neighbors = torch.cat([neighbor_info[i]['user'] for i in range(len(neighbor_info))], dim=0).long()
        pad_neighbors_time = torch.cat([neighbor_info[i]['time'] for i in range(len(neighbor_info))], dim=0)
        pad_neighbors_freqbin = torch.cat([neighbor_info[i]['freqbin'] for i in range(len(neighbor_info))], dim=0)

        return all_user_info, cand_items, labels, item_freqbins, pad_neighbors, pad_neighbors_time, pad_neighbors_freqbin

    

class DataLoader: 
    def __init__(self, opt):
        self.opt = opt
        self.dpath = opt.dataset_path + '/'
        self.bs = opt.batch_size
        self.trn_numneg = opt.numneg
        
        self.trn_loader, self.vld_loader, self.tst_loader = self.get_loaders_for_seqrs()        
    
        print(("train/val/test/ divided by batch size {:d}/{:d}/{:d}".format(
                    len(self.trn_loader), len(self.vld_loader),len(self.tst_loader))))
        print("=" * 80)
        
    def get_loaders_for_seqrs(self):
        print("\n📋 Loading data...\n")
        trn_loader = self.get_each_loader(self.dpath+'trn', self.bs, self.trn_numneg, shuffle=True)
        vld_loader = self.get_each_loader(self.dpath+'vld', int(self.bs/2), self.trn_numneg, shuffle=False)
        tst_loader = self.get_each_loader(self.dpath+'tst', int(self.bs/2), self.trn_numneg, shuffle=False)    
        
        return trn_loader, vld_loader, tst_loader
    
    def get_each_loader(self, data_path, batch_size, trn_negnum, shuffle=True, num_workers=0):
        """Builds and returns Dataloader."""

        dataset = SEQRS_Dataset(data_path, trn_negnum, self.opt)

        if data_path.endswith('trn'):
            collate = dataset.train_collate
        else:
            collate = dataset.test_collate

        data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate)

        return data_loader
    
    def get_loaders(self):
        return self.trn_loader, self.vld_loader, self.tst_loader
    
    def get_embedding(self):
        return self.input_embedding
            

    