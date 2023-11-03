
import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel
from google_drive_downloader import GoogleDriveDownloader as gdd

from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation

from model.espnet import *

# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places'  : '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places' : '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2'   : '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2'  : '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2'      : '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)

        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x

class GraphVLAD(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)

        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim
            
        # Semantic Segmentation
        self.classes = 20
        self.p = 2
        self.q = 8
        # self.encoderFile = "/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth"
        self.Espnet = ESPNet(args, classes=self.classes, p=self.p, q = self.q)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        # requires_grad=False
        
                
        #graph
        self.input_dim = 4096 # 16384# 8192
        self.hidden_dim = [2048,4096]#[8192, 8192]
        self.num_neighbors_list = [5]#,2]
        
        self.graph = aggregation.GraphSage(input_dim=self.input_dim, hidden_dim=self.hidden_dim, 
                                           num_neighbors_list=self.num_neighbors_list)
        
    # def _init_params(self):
    #     self.aggregation.initialize_netvlad_layer() 
        
    def forward(self, x):
    
        # fixing for Tokyo
        sizeH = x.shape[2]
        sizeW = x.shape[3]
        
        if sizeH%2 != 0:
            x = F.pad(input=x, pad=(0,0,1,2), mode='constant', value=0)
        if sizeW%2 != 0:
            x = F.pad(input=x, pad=(1,2), mode='constant', value=0)
        
        # createboxes
        # print("debuggind started")
        with torch.no_grad():
            b_out = self.Espnet(x)
        # b_out = self.Espnet(x)
        mask = b_out.max(1)[1]   #torch.Size([36, 480, 640])
        
        for jj in range(len(mask)):  #batch processing
            
            single_label_mask = mask[jj]
            
            # obj_ids = torch.unique(single_label_mask)
            obj_ids, obj_i = single_label_mask.unique(return_counts=True)
            obj_ids = obj_ids[1:] 
            obj_i = obj_i[1:]
            #torch.Size([19])
            
            masks = single_label_mask == obj_ids[:, None, None]
            boxes_t = torchvision.ops.masks_to_boxes(masks.to(torch.float32))

            
            rr_boxes = torch.argsort(torch.argsort(obj_i,descending=True)) # (decending order)

    
            
            boxes = boxes_t/16
        
        # print(x.shape)
        _, _, H, W = x.shape
        patch_mask = torch.zeros((H, W)).cuda()
        
        # Running Backbone
        x = self.backbone(x)
        
        N, C, H, W = x.shape

        
        # img_orig = to_tensor(Image.open(image_list[jj]).convert('RGB'))
        # _, H, W = img_orig.shape

        bb_x = [[int(W/4), int(H/4), int(3*W/4),int(3*H/4)],
                [0, 0, int(W/3),H], 
                [0, 0, W,int(H/3)], 
                [int(2*W/3), 0, W,H], 
                [0, int(2*H/3), W,H]]

 
        NB = 5
        
        graph_nodes = torch.zeros(N,NB,C,H,W).cuda()
        rsizet = torchvision.transforms.Resize((H,W)) #H W

        
        for Nx in range(N):    
            # img_stk = x[Nx].unsqueeze(0)
            img_nodes = []
            # print(Nx)
            for idx in range(len(boxes)):
                for b_idx in range(len(rr_boxes)):


                    if idx == rr_boxes[b_idx] and obj_i[b_idx] > 10000 and len(img_nodes) < NB-1:

                        patch_mask = patch_mask*0

                        # label obj_ids[rr_boxes[b_idx]]
                        patch_mask[single_label_mask == obj_ids[b_idx]] = 1
                        # box boxes[rr_boxes[b_idx]]

                        # patch_mask = patch_mask.unsqueeze(0)
                        patch_maskr = rsizet(patch_mask.unsqueeze(0))
                        
                        patch_maskr = patch_maskr.squeeze(0)

                        boxesd = boxes.to(torch.long)
                        x_min,y_min,x_max,y_max = boxesd[b_idx]
                    

                        c_img = x[Nx][:, y_min:y_max,x_min:x_max]
                        

                        resultant = rsizet(c_img)
 
                        img_nodes.append(resultant.unsqueeze(0))
                        
                        break                    
            

            if len(img_nodes) < NB:
                for i in range(len(bb_x)-len(img_nodes)):
                    x_cropped =  x[Nx][: ,bb_x[i][1]:bb_x[i][3], bb_x[i][0]:bb_x[i][2]]
                    img_nodes.append(rsizet(x_cropped.unsqueeze(0)))
                    
                
        
            aa = torch.stack(img_nodes,1)
            # code.interact(local=locals())
            graph_nodes[Nx] = aa[0]

        
        
        node_features_list = []
        neighborsFeat = []

        x_cropped = graph_nodes.view(NB,N,C,H,W)

        x_cropped = torch.cat((graph_nodes.view(NB,N,C,H,W), x.unsqueeze(0)))
         
        for i in range(NB+1):
            
            
            vlad_x = self.aggregation(x_cropped[i])
            
            # [IMPORTANT] normalize
            # vlad_x1 = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
            # vlad_x = vlad_x.view(x.size(0), -1)  # flatten
            # vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize
            # aa = vlad_x.shape #32, 32768
            #vlad_x = vlad_x.view(-1,8192) # 8192
            # print(i)
            
            neighborsFeat.append(vlad_x)

        #code.interact(local=locals())
        node_features_list.append(neighborsFeat[NB])
        node_features_list.append(torch.concat(neighborsFeat[0:NB],0))
        # code.interact(local=locals())
        
        
        
        neighborsFeat = []
        #vlad_x = []
        
        ## Graphsage
        x = self.graph(node_features_list)

        x = torch.add(x,vlad_x)

        
        return x.view(-1,32768)
        
        
        
        # x = self.aggregation(x)
        # return x

def get_aggregation(args):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                   work_with_tokens=args.work_with_tokens)
    elif args.aggregation == 'crn':
        return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "rrm":
        return aggregation.RRM(args.features_dim)
    elif args.aggregation in ['cls', 'seqpool']:
        return nn.Identity()


def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name +".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name],
                                            dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_backbone(args):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = args.backbone.startswith('cct') or args.backbone.startswith('vit')
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        for name, child in backbone.named_children():
            # Freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        if args.backbone.endswith("conv4"):
            logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
        if args.pretrain == 'offtheshelf' :
            model_name = "vd16_offtheshelf_conv5_3_max"
            file_path = join("data", "pretrained_nets", model_name +".pth")
            backbone.load_state_dict(torch.load(file_path),strict=False)
            print("Matconvnet Loaded")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vit"):
        assert args.resize[0] in [224, 384], f'Image size for ViT must be either 224 or 384, but it\'s {args.resize[0]}'
        if args.resize[0] == 224:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.resize[0] == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        backbone = VitWrapper(backbone, args.aggregation)
                
        args.features_dim = 768
        return backbone

    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone
class VitWrapper(nn.Module):
    def __init__(self, vit_model, aggregation):
        super().__init__()
        self.vit_model = vit_model
        self.aggregation = aggregation
    def forward(self, x):
        if self.aggregation in ["netvlad", "gem"]:
            return self.vit_model(x).last_hidden_state[:, 1:, :]
        else:
            return self.vit_model(x).last_hidden_state[:, 0, :]


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]

