import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AlexSal(nn.Module):
    def __init__(self, placesmodel_path):
        super(AlexSal, self).__init__()

        self.features = nn.Sequential(
            *list(torch.load(placesmodel_path).features.children())[:-2]
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv6 = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.relu(self.features(x))
        x = self.relu(self.conv6(x))
        x = x.squeeze(1)
        return x

class AlexGaze(nn.Module):
    def __init__(self):
        super(AlexGaze, self).__init__()
        self.features = nn.Sequential(
            *list(models.alexnet(pretrained=True).features.children())
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(9216, 500)
        self.fc2 = nn.Linear(669, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 169)

        self.finalconv = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x, egrid):
        x = self.relu(self.features(x))
        x = x.view(-1, 9216)
        x = self.relu(self.fc1(x))

        egrid = egrid.view(-1, 169)
        egrid = egrid * 24

        x = torch.cat((x, egrid), dim=1)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))

        x = x.view(-1, 1, 13, 13)
        x = self.finalconv(x)
        x = x.squeeze(1)
        return x


class GazeNet(nn.Module):
    def __init__(self, placesmodel_path='./alexnet_places365.pth'):
        super(GazeNet, self).__init__()
        self.salpath = AlexSal(placesmodel_path)
        self.gazepath = AlexGaze()

        self.smax = nn.LogSoftmax(dim=1)
        self.nolog_smax = nn.Softmax(dim=1)

        self.fc_0_0 = nn.Linear(169, 25)
        self.fc_0_m1 = nn.Linear(169, 25)
        self.fc_0_1 = nn.Linear(169, 25)
        self.fc_m1_0 = nn.Linear(169, 25)
        self.fc_1_0 = nn.Linear(169, 25)

    def forward(self, xi, xh, xp):

        outxi = self.salpath(xi)
        outxh = self.gazepath(xh, xp)
        output = outxi * outxh
        output = output.view(-1, 169)

        out_0_0 = self.smax(self.fc_0_0(output))
        out_1_0 = self.smax(self.fc_1_0(output))
        out_m1_0 = self.smax(self.fc_m1_0(output))
        out_0_m1 = self.smax(self.fc_0_m1(output))
        out_0_1 = self.smax(self.fc_0_1(output))

        return [out_0_0, out_1_0, out_m1_0, out_0_m1, out_0_1]


    def predict(self, xi, xh, xp):

        outxi = self.salpath(xi)
        outxh = self.gazepath(xh, xp)
        output = outxi * outxh
        output = output.view(-1, 169)

        hm = torch.zeros(output.size(0), 15, 15).cuda()
        count_hm = torch.zeros(output.size(0), 15, 15).cuda()

        f_0_0 = self.nolog_smax(self.fc_0_0(output)).view(-1, 5, 5)
        f_1_0 = self.nolog_smax(self.fc_1_0(output)).view(-1, 5, 5)
        f_m1_0 = self.nolog_smax(self.fc_m1_0(output)).view(-1, 5, 5)
        f_0_m1 = self.nolog_smax(self.fc_0_m1(output)).view(-1, 5, 5)
        f_0_1 = self.nolog_smax(self.fc_0_1(output)).view(-1, 5, 5)

        f_cell = []
        f_cell.extend([f_0_m1, f_0_1, f_m1_0, f_1_0, f_0_0])

        v_x = [0, 1, -1, 0, 0]
        v_y = [0, 0, 0, -1, 1]

        for k in range(5):
            dx, dy = v_x[k], v_y[k]
            f = f_cell[k]
            for x in range(5):
                for y in range(5):

                    i_x = 3*x - dx
                    i_x = max(i_x, 0)
                    if x == 0:
                        i_x = 0

                    i_y = 3*y - dy
                    i_y = max(i_y, 0)
                    if y == 0:
                        i_y = 0

                    f_x = 3*x + 2 - dx
                    f_x = min(14, f_x)
                    if x == 4:
                        f_x = 14

                    f_y = 3*y + 2 - dy
                    f_y = min(14, f_y)
                    if y == 4:
                        f_y = 14

                    a = f[:, x, y].contiguous()
                    a = a.view(output.size(0), 1, 1)

                    hm[:, i_x: f_x+1, i_y: f_y+1] =  hm[:, i_x: f_x+1, i_y: f_y+1] + a
                    count_hm[:, i_x: f_x+1, i_y: f_y+1] = count_hm[:, i_x: f_x+1, i_y: f_y+1] + 1

        hm_base = hm.div(count_hm)

        hm_base = hm_base.unsqueeze(1)

        hm_base = F.interpolate(input = hm_base, size = (227, 227), mode='bicubic', align_corners=False)

        hm_base = hm_base.squeeze(1)


        #modeltester works with this return:
        #return hm_base

        #main.py /training must use this
        return hm_base.view(-1, 227 * 227)

    def raw_hm(self, xi, xh, xp):

        outxi = self.salpath(xi)
        outxh = self.gazepath(xh, xp)
        output = outxi * outxh
        output = output.view(-1, 169)

        hm = torch.zeros(output.size(0), 15, 15).cuda()
        count_hm = torch.zeros(output.size(0), 15, 15).cuda()

        f_0_0 = self.nolog_smax(self.fc_0_0(output)).view(-1, 5, 5)
        f_1_0 = self.nolog_smax(self.fc_1_0(output)).view(-1, 5, 5)
        f_m1_0 = self.nolog_smax(self.fc_m1_0(output)).view(-1, 5, 5)
        f_0_m1 = self.nolog_smax(self.fc_0_m1(output)).view(-1, 5, 5)
        f_0_1 = self.nolog_smax(self.fc_0_1(output)).view(-1, 5, 5)

        f_cell = []
        f_cell.extend([f_0_m1, f_0_1, f_m1_0, f_1_0, f_0_0])

        v_x = [0, 1, -1, 0, 0]
        v_y = [0, 0, 0, -1, 1]

        for k in range(5):
            dx, dy = v_x[k], v_y[k]
            f = f_cell[k]
            for x in range(5):
                for y in range(5):

                    i_x = 3*x - dx
                    i_x = max(i_x, 0)
                    if x == 0:
                        i_x = 0

                    i_y = 3*y - dy
                    i_y = max(i_y, 0)
                    if y == 0:
                        i_y = 0

                    f_x = 3*x + 2 - dx
                    f_x = min(14, f_x)
                    if x == 4:
                        f_x = 14

                    f_y = 3*y + 2 - dy
                    f_y = min(14, f_y)
                    if y == 4:
                        f_y = 14

                    a = f[:, x, y].contiguous()
                    a = a.view(output.size(0), 1, 1)

                    hm[:, i_x: f_x+1, i_y: f_y+1] =  hm[:, i_x: f_x+1, i_y: f_y+1] + a
                    count_hm[:, i_x: f_x+1, i_y: f_y+1] = count_hm[:, i_x: f_x+1, i_y: f_y+1] + 1

        hm_base = hm.div(count_hm)
        raw_hm = hm_base

        hm_base = hm_base.unsqueeze(1)

        hm_base = F.interpolate(input = hm_base, size = (227, 227), mode='bicubic', align_corners=False)

        hm_base = hm_base.squeeze(1)


        #this one is size (-1, 227, 227)
        #return hm_base

        #main.py /training must use this
        return hm_base.view(-1, 227 * 227), raw_hm
