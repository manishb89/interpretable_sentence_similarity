import torch
from torch.autograd import Variable

class C1(torch.nn.Module):
    def __init__(self, cfg=None):
        super(C1, self).__init__()
        self.cfg = cfg
        
    def get_rel_mask(self, data_point):
        rel_mask = torch.zeros(data_point['left_chunks'].shape[0], data_point['right_chunks'].shape[0])

        rels = data_point['constr']
        for rel in rels:
            s1_idx, s2_idx = rel[0], rel[1]
            rel_mask[s1_idx][s2_idx] = 1.0
        
        rel_mask = Variable(rel_mask, requires_grad=False)

        return rel_mask

    def forward(self, data_point):
        mask = self.get_rel_mask(data_point)
        # if opt.gpuid != -1:
        #     mask = mask.cuda()
        return mask
