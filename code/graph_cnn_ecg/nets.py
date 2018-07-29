from all_imports import *
from sklearn.metrics import *


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