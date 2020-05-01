import torch
import torch.nn as NN
from torch.autograd import Variable
from torch.nn import Parameter
from constraints import C1 as C1


def sinkhorn_knopp(activations, r, c):
    '''
    @param activations: pointer network activations
    @param r: row sums
    @param c: column sums
    '''
    device = activations.device
    activations = - activations
    row_adjust = torch.min(activations, 1)[0]
    activations = activations - row_adjust.reshape((-1, 1))
    activations = activations + 1e-2

    P = torch.exp(-0.6 * activations).double().to(device)
    P = P / P.sum()
    u = torch.zeros(activations.shape[0]).double().to(device)

    itr_idx = 0
    while torch.max(torch.abs(u - P.sum(1))) > 1e-5:
        u = P.sum(1)
        P = P * (r / u).reshape((-1, 1))
        P = P * (c / P.sum(0)).reshape((1, -1))
        itr_idx += 1
        if itr_idx == 100: break
    return P.double()


def masked_log_softmax(data, mask):
    data = torch.exp(data) * mask
    data_softmax = data / (torch.sum(data, 1).unsqueeze(1) + 1e-15)
    return torch.log(data_softmax)


class PointerNetwork(NN.Module):

    def __init__(self, cfg):
        super(PointerNetwork, self).__init__()
        self.cfg = cfg

        if torch.cuda.is_available() and cfg.has_key('gpuid'):
            self.device = torch.device('cuda:{}'.format(self.cfg.gpuid))
        else:
            self.device = torch.device('cpu')

        self.hidden_dim = self.cfg.hidden_dim
        self.input_dim = self.cfg.input_dim

        self.w1 = Parameter(torch.zeros(self.hidden_dim, self.input_dim).to(self.device),
                            requires_grad=True)
        self.w1 = torch.nn.init.xavier_normal_(self.w1)

        self.w2 = Parameter(torch.zeros(self.hidden_dim, self.input_dim).to(self.device),
                            requires_grad=True)
        self.w2 = torch.nn.init.xavier_normal_(self.w2)

        self.w3 = Parameter(torch.zeros(self.hidden_dim, self.input_dim).to(self.device),
                            requires_grad=True)
        self.w3 = torch.nn.init.xavier_normal_(self.w3)

        self.v = Parameter(torch.zeros(1, self.hidden_dim).to(self.device), requires_grad=True)
        self.v = torch.nn.init.xavier_normal_(self.v)

        self.phi = Parameter(torch.zeros(1, self.input_dim).to(self.device), requires_grad=True)

        self.phi = torch.nn.init.xavier_normal_(self.phi)

        self.b1 = Parameter(torch.ones(1).to(self.device), requires_grad=True)
        self.b2 = Parameter(torch.zeros(1).to(self.device), requires_grad=True)

        self.c1 = Parameter(torch.ones(1).to(self.device), requires_grad=True)
        self.c2 = Parameter(torch.zeros(1).to(self.device), requires_grad=True)

        self.c3 = Parameter(torch.ones(1).to(self.device), requires_grad=True)
        self.c4 = Parameter(torch.ones(1).to(self.device), requires_grad=True)

        self.constr_layers = []
        for constr in cfg.output_constr.split(','):
            if constr == 'C1':
                self.constr_layers.append(C1.C1(self.cfg))
            elif constr == '':
                pass
            else:
                print 'Unknown constraint: {}'.format(constr)
        if self.constr_layers != []:
            print 'FOL constraints enabled'

    def forward_debug(self, inputs):
        lx, rx = inputs["left_embedding"], inputs["right_embedding"]
        left_l, right_l = inputs["num_left_chunks"], inputs["num_right_chunks"]

        b = torch.matmul(self.v, torch.tanh(torch.matmul(self.w1, lx.transpose(1, 0))
                                            + torch.matmul(self.w2, self.phi.transpose(1, 0))))
        b = torch.squeeze(b)

        b_tilda = torch.matmul(self.v, torch.tanh(torch.matmul(self.w1, self.phi.transpose(1, 0))
                                                  + torch.matmul(self.w2, rx.transpose(1, 0))))
        b_tilda = torch.squeeze(b_tilda)
        rx = rx.transpose(1, 0)

        if self.constr_layers != []:
            all_constr_masks = []
            for layer in self.constr_layers:
                all_constr_masks.append(
                    layer(inputs).unsqueeze(0))  # unsqueeze for aggregating scores from all constraints
            constr_mask = torch.cat(all_constr_masks, 0).sum(0)

        activations = []
        for i, li in enumerate(lx):
            li_w = torch.matmul(self.w1, li)
            rx_w = torch.matmul(self.w2, rx)
            activations_li = torch.tanh(torch.matmul(self.w1, li)[:, None]
                                        + torch.matmul(self.w2, rx)
                                        )
            activations_li = torch.squeeze(torch.matmul(self.v, activations_li))
            activations_li[right_l:] = 0
            # activations_li = torch.log_softmax(activations_li, dim=0)
            # Constraints
            if self.constr_layers != []:
                activations_li = activations_li + self.cfg.rho * constr_mask[i]

            # activations_li = torch.log_softmax(activations_li, dim=0)
            activations.append(activations_li.unsqueeze(dim=0))

        activations = torch.cat(activations)

        g = activations
        g_mask = torch.zeros(activations.shape).double()
        g_mask[:left_l, :right_l] = 1
        g = g * g_mask.float()
        g = (torch.max(g[:, :right_l], 1)[0])
        g = torch.sigmoid(g)

        activations_t = activations.transpose(1, 0)
        g_tilda = activations_t
        g_mask = torch.zeros(activations.shape)
        g_mask[:right_l, :left_l] = 1
        g_tilda = g_tilda * g_mask.float()
        g_tilda = (torch.max(g_tilda[:, :left_l], 1)[0])
        g_tilda = torch.sigmoid(g_tilda)

        activations_mask = torch.ones(activations.shape)
        activations_mask[left_l:, :-1] = -50.0
        activations_mask[:, right_l:-1] = -50.0
        activations = activations * activations_mask
        log_activations = torch.log_softmax(activations, 1)
        log_activations = log_activations + torch.log(g).unsqueeze(1)
        log_activations = torch.cat([log_activations, torch.unsqueeze(torch.log(1 - g + 1e-15), 1)], dim=1).double()
        return torch.exp(log_activations), g, g_tilda

    def forward(self, inputs):
        '''
        @param inputs: input data for left and right sentences including embeddings, num_chunks and constraints
        '''
        lx, rx = inputs["left_embedding"], inputs["right_embedding"]
        left_l, right_l = inputs["num_left_chunks"], inputs["num_right_chunks"]

        b = torch.matmul(self.v, torch.tanh(torch.matmul(self.w1, lx.transpose(1, 0))
                                            + torch.matmul(self.w2, self.phi.transpose(1, 0))))
        b = torch.squeeze(b)

        b_tilda = torch.matmul(self.v, torch.tanh(torch.matmul(self.w1, self.phi.transpose(1, 0))
                                                  + torch.matmul(self.w2, rx.transpose(1, 0))))
        b_tilda = torch.squeeze(b_tilda)
        rx = rx.transpose(1, 0)

        if self.constr_layers != []:
            all_constr_masks = []
            for layer in self.constr_layers:
                all_constr_masks.append(
                    layer(inputs).unsqueeze(0))  # unsqueeze for aggregating scores from all constraints
            constr_mask = torch.cat(all_constr_masks, 0).sum(0).to(self.device)
            rho = Variable(torch.Tensor([self.cfg.rho]).to(self.device), requires_grad=False)

        activations = []
        rx_o = rx.transpose(1, 0)

        for i, li in enumerate(lx):
            hadamard = (li * rx_o).transpose(1, 0)
            hd_w = torch.matmul(self.w3, hadamard)

            activations_li = torch.tanh(torch.matmul(self.w1, li)[:, None]
                                        + torch.matmul(self.w2, rx)
                                        + hd_w
                                        )
            activations_li = torch.squeeze(torch.matmul(self.v, activations_li))
            activations_li[right_l:] = 0
            activations.append(activations_li.unsqueeze(dim=0))

        activations = torch.cat(activations)
        if "syn_scores" in inputs:
            activations = activations + (6.0 - torch.relu(self.c4)) * inputs["syn_scores"]

        for i, _ in enumerate(lx):
            if self.constr_layers != []:
                activations[i] = activations[i] + rho * constr_mask[i]

        g = activations
        g_mask = torch.zeros(activations.shape).double().to(self.device)
        g_mask[:left_l, :right_l] = 1
        g = g * g_mask.float()
        g = torch.max(g[:, :right_l], 1)[0]  # + torch.max(activations[:, :right_l], 1)[0])
        g = torch.sigmoid(self.b1 * g + self.b2)

        activations_t = activations.transpose(1, 0)

        g_tilda = activations_t
        g_mask = torch.zeros(activations.shape).to(self.device)
        g_mask[:right_l, :left_l] = 1
        g_tilda = g_tilda * g_mask.float()
        g_tilda = torch.max(g_tilda[:, :left_l], 1)[0]  # + torch.max(activations_t[:, :left_l], 1)[0])
        g_tilda = torch.sigmoid(self.c1 * g_tilda + self.c2)

        a = g_tilda * activations
        a_mask = torch.zeros(activations.shape).to(self.device)
        a_mask[:left_l, :right_l] = 1
        a = a * a_mask.float()
        activations = a

        activations_mask = torch.ones(activations.shape).to(self.device)

        activations_mask[left_l:, :-1] = -50.0
        activations_mask[:, right_l:-1] = -50.0
        activations = activations * activations_mask

        g_r = torch.ones(g.shape[0]).double().to(self.device)
        g_tilda_c = torch.ones(g_tilda.shape[0]).double().to(self.device)

        log_activations = torch.log(sinkhorn_knopp(activations, g_r, g_tilda_c)).double()
        log_activations = log_activations + torch.log(g).unsqueeze(1).double()
        log_activations = torch.cat([log_activations, torch.unsqueeze(torch.log(1 - g + 1e-15).double(), 1)],
                                    dim=1).double()
        return log_activations, torch.log(1 - g_tilda + 1e-15)
