from all_imports import *
from sklearn.metrics import *

class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        # pdb.set_trace()
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx

class gconv1d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kern = kernel_size
#         if self.kern <= 0:
#             self.kern = 1
#         self.lin = torch.nn.Linear(in_c * self.kern, out_c)
        self.lin = torch.nn.Linear(in_c * self.kern, out_c)
        return

    def forward(self, inp, L):
#         import pdb; pdb.set_trace()
        B, V, Fin = inp.shape
        x0 = inp.permute(1, 2, 0).contiguous()
        x0 = x0.view([V, Fin*B])
        x = x0.unsqueeze(0)
        L = L[0]
        if self.kern > 1:
            x1 = my_sparse_mm()(L, x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for k in range(2, self.kern):
            x2 = 2 * my_sparse_mm()(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2

        x = x.view([self.kern, V, Fin, B])           # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B*V, Fin*self.kern])             # B*V x Fin*K
#         pdb.set_trace()
        try:
            x = self.lin(x)
        except:
            self.lin = self.lin.cuda()
            try:
                x = self.lin(x)
            except:
                pdb.set_trace()
        x = x.view([B, V, self.out_c])             # B x V x Fout
        return x

def get_scores(learner):
    pred, targs = learner.TTA()
    probs = np.mean(np.exp(pred),0)
    preds = np.argmax(probs,axis=1)
    
    ac = accuracy_np(probs,targs)
    f1 = f1_score(targs,preds,average=None)
    prec = precision_score(targs,preds,average=None)
    rec = recall_score(targs,preds,average=None)
    print('Ac', ac)
    print('f1', f1)
    print('prec', prec)
    print('rec', rec)
    return pred,targs,ac,f1,prec,rec

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv1d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out
    
    
class  senet_small(nn.Module):
    def __init__(self, block, inc_list, inc_scale, num_blocks_list, stride_list, num_classes):
        super().__init__()
        self.num_blocks = len(num_blocks_list)
        inc_list1 = [o//inc_scale for o in inc_list]
        self.in_planes = inc_list1[0]
        self.conv1 = nn.Conv1d(15, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_planes)
        
        lyrs = []
        for inc, nb, strl in zip(inc_list1[1:], num_blocks_list, stride_list):
            lyrs.append(self._make_layer(block, inc, nb, strl))
            
        self.lyrs = nn.Sequential(*lyrs)
        self.linear = nn.Linear(inc_list1[-1], num_classes)
        
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, inp):
#         import pdb; pdb.set_trace()
        out = F.relu(self.bn1(self.conv1(inp)))
        out = self.lyrs(out)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)
    
def get_senet_small():
    snt_sml = senet_small(PreActBlock, 
                    inc_list=[64, 64, 128, 256], 
                    inc_scale = 4,
                    num_blocks_list=[2, 3, 2], 
                    stride_list=[1, 2, 2], 
                    num_classes=2)
    return snt_sml


class GraphPreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
#         self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = gconv1d(in_planes, planes, kernel_size=3)
#         self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = gconv1d(planes, planes, kernel_size=3)

        # SE layers
        self.fc1 = gconv1d(planes, planes//16, kernel_size=1)
        self.fc2 = gconv1d(planes//16, planes, kernel_size=1)

    def forward(self, x, L):
        out = F.relu(x)
        shortcut = x
        out = self.conv1(out, L)
        out = self.conv2(F.relu(out), L)

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w, L))
        w = F.sigmoid(self.fc2(w, L))
        # Excitation
        out = out * w

        out += shortcut
        return out
    
    
class  graph_senet_small(nn.Module):
    def __init__(self, block, inc_list, inc_scale, num_blocks_list, stride_list, num_classes):
        super().__init__()
        self.num_blocks = len(num_blocks_list)
        inc_list1 = [o//inc_scale for o in inc_list]
        self.in_planes = inc_list1[0]
        self.conv1 = gconv1d(149, self.in_planes, kernel_size=3)
#         self.bn1 = nn.BatchNorm1d(self.in_planes)
        
        lyrs = []
        for inc, nb, strl in zip(inc_list1[1:], num_blocks_list, stride_list):
            lyrs.append(self._make_layer(block, inc, nb, strl))
            
        self.lyrs = lyrs
#         self.lyrs = nn.Sequential(*lyrs)
        self.linear = nn.Linear(inc_list1[-1], num_classes)
        
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return layers
    
    def forward(self, inp, L):
#         import pdb; pdb.set_trace()
        out = F.relu(self.conv1(inp, L))
#         out = self.lyrs(out, L)
        for lyr in self.lyrs:
            for ly in lyr:
                out = ly(out, L)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)
    
def get_graph_senet_small():
    snt_sml = graph_senet_small(GraphPreActBlock, 
                        inc_list=[64, 64, 128, 256], 
                        inc_scale = 4,
                        num_blocks_list=[2, 3, 2], 
                        stride_list=[1, 1, 1], 
                        num_classes=2)
    return snt_sml