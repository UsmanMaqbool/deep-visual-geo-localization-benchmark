
import math
import torch
import torch.nn.functional as F

def get_loss(self, vlad_encoding, loss_type, B, N, nNeg):
    
    # if B*N!=vlad_encoding.shape[0]:
        # vlad_encoding = vlad_encoding[:B*N,:]
    # print("vlad_encoding: " , vlad_encoding.shape)

    # outputs = vlad_encoding.view(B, N, -1)
    # print("outputs: " , outputs.shape)

    #N: 12
    # B: 1
    # nNeg: 10

    temp = 0.07
    

    # outputs = outputs.view(B, N, -1)

    outputs = vlad_encoding.view(B, N, -1)
    L = vlad_encoding.size(-1)

    output_negatives = outputs[:, 2:]
    output_anchors = outputs[:, 0]
    output_positives = outputs[:, 1]

    # output_anchors, output_positives, output_negatives = torch.split(vlad_encoding, [B, B, nNeg])
    
    if (loss_type=='triplet'):
        output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
        output_positives = output_positives.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
        output_negatives = output_negatives.contiguous().view(-1, L)
        loss = F.triplet_margin_loss(output_anchors, output_positives, output_negatives,
                                        margin=self.margin, p=2, reduction='mean')
        #cself.margin**0.5
    elif (loss_type=='sare_joint'):
        # ### original version: euclidean distance
        # dist_pos = ((output_anchors - output_positives)**2).sum(1)
        # dist_pos = dist_pos.view(B, 1)

        # output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
        # output_negatives = output_negatives.contiguous().view(-1, L)
        # dist_neg = ((output_anchors - output_negatives)**2).sum(1)
        # dist_neg = dist_neg.view(B, -1)

        # dist = - torch.cat((dist_pos, dist_neg), 1)
        # dist = F.log_softmax(dist, 1)
        # loss = (- dist[:, 0]).mean()

        ## new version: dot product
        dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
        dist_pos = dist_pos.diagonal(0)
        dist_pos = dist_pos.view(B, 1)
        
        output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
        output_negatives = output_negatives.contiguous().view(-1, L)
        dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
        dist_neg = dist_neg.diagonal(0)
        dist_neg = dist_neg.view(B, -1)
        
        dist = torch.cat((dist_pos, dist_neg), 1)/temp
        dist = F.log_softmax(dist, 1)
        loss = (- dist[:, 0]).mean()

    elif (loss_type=='sare_ind'):
        # ### original version: euclidean distance
        # dist_pos = ((output_anchors - output_positives)**2).sum(1)
        # dist_pos = dist_pos.view(B, 1)

        # output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
        # output_negatives = output_negatives.contiguous().view(-1, L)
        # dist_neg = ((output_anchors - output_negatives)**2).sum(1)
        # dist_neg = dist_neg.view(B, -1)

        # dist_neg = dist_neg.unsqueeze(2)
        # dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
        # dist = - torch.cat((dist_pos, dist_neg), 2).view(-1, 2)
        # dist = F.log_softmax(dist, 1)
        # loss = (- dist[:, 0]).mean()

        ## new version: dot product
        dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
        dist_pos = dist_pos.diagonal(0)
        dist_pos = dist_pos.view(B, 1)
        
        output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
        output_negatives = output_negatives.contiguous().view(-1, L)
        dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
        dist_neg = dist_neg.diagonal(0)
        dist_neg = dist_neg.view(B, -1)
        
        dist_neg = dist_neg.unsqueeze(2)
        dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
        dist = torch.cat((dist_pos, dist_neg), 2).view(-1, 2)/temp
        dist = F.log_softmax(dist, 1)
        loss = (- dist[:, 0]).mean()

    else:
        assert ("Unknown loss function")

    return loss   

def mac(x):
    return F.adaptive_max_pool2d(x, (1,1))

def spoc(x):
    return F.adaptive_avg_pool2d(x, (1,1))

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

def rmac(x, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension
    W = x.size(3)
    H = x.size(2)
    w = min(W, H)
    # w2 = math.floor(w/2.0 - 1)
    b = (max(H, W)-w)/(steps-1)
    (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension
    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1
    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)
    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)
        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
                R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt
    return v

