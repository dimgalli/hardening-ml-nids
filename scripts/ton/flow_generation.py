import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import ARGVA, GCNConv


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, z, edge_index, sigmoid=True):
        z = self.conv1(z, edge_index)
        z = self.conv2(z, edge_index)
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


def train():
    model.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    for i in range(5):
        discriminator_optimizer.zero_grad()
        discriminator_loss = model.discriminator_loss(z)
        discriminator_loss.backward()
        discriminator_optimizer.step()
    recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)
    reg_loss = model.reg_loss(z)
    kl_loss = (1 / train_data.num_nodes) * model.kl_loss()
    loss = recon_loss + reg_loss + kl_loss
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    z = model.encode(test_data.x, test_data.edge_index)
    auc, ap = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
    return auc, ap


parser = argparse.ArgumentParser(description='Generate synthetic data')
parser.add_argument('--processed-data', type=str, required=True, help='path to processed data')
parser.add_argument('--generated-data', type=str, required=True, help='path to save the generated data')

args = parser.parse_args()

if not os.path.exists(args.processed_data) or not os.path.isfile(args.processed_data):
    sys.exit('Path to processed data does not exist or is not a file')

data = pd.read_csv(args.processed_data)

endp = data.iloc[:, 0:2]

srcnodes = np.unique(endp.src_ip_port)
dstnodes = np.unique(endp.dst_ip_port)
nodes = np.union1d(srcnodes, dstnodes)

d = {node: n for n, node in enumerate(nodes)}

edge_index = np.transpose([[d[srcnode], d[dstnode]] for srcnode, dstnode in endp.values])
edge_index = torch.from_numpy(edge_index.astype(np.int64))

feat = data.iloc[:, 2:39]

scaler = MinMaxScaler()

edge_attr = scaler.fit_transform(feat.values)
edge_attr = torch.from_numpy(edge_attr.astype(np.float32))

labels = data.iloc[:, 39]

y = labels.values
y = torch.from_numpy(y.astype(np.int64))

data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y)

transform = T.LineGraph()
data = transform(data)

transform = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=False)
train_data, val_data, test_data = transform(data)

encoder = Encoder(in_channels=data.num_features, hidden_channels=16, out_channels=16)
discriminator = Discriminator(in_channels=16, hidden_channels=32, out_channels=1)
decoder = Decoder(in_channels=16, hidden_channels=data.num_features, out_channels=data.num_features)

model = ARGVA(encoder, discriminator, decoder)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.005)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.005)

for epoch in range(1, 201):
    loss = train()
    auc, ap = test()
    print('Epoch {:03d}, Loss = {:.3f}, AUC = {:.3f}, AP = {:.3f}'.format(epoch, loss, auc, ap))

z = model.encode(data.x, data.edge_index)

regressor = RandomForestRegressor(n_estimators=10, n_jobs=-1)
regressor.fit(z.detach().numpy(), data.x.detach().numpy())
x = regressor.predict(z.detach().numpy())

x = pd.DataFrame(scaler.inverse_transform(x), columns=feat.columns)

classifier = RandomForestClassifier(n_estimators=10, n_jobs=-1)
classifier.fit(z.detach().numpy(), data.y.detach().numpy())
y = classifier.predict(z.detach().numpy())

y = pd.Series(y, name=labels.name)

data = pd.concat([x, y], axis=1)

data[data.columns.drop('duration')] = data[data.columns.drop('duration')].round().astype(int)

data = data.drop(data[~data['ip_src_type'].isin([0, 1])].index)

data = data.drop(data[~data['ip_dst_type'].isin([0, 1])].index)

data = data.drop(data[data['src_port_wellknown'] + data['src_port_registered'] + data['src_port_private'] != 1].index)

data = data.drop(data[data['dst_port_wellknown'] + data['dst_port_registered'] + data['dst_port_private'] != 1].index)

data = data.drop(data[data['tcp'] + data['udp'] != 1].index)

data = data.drop(data[data['-'] + data['dns'] + data['http'] + data['ssl'] != 1].index)

data = data.drop(data[data['S0'] + data['S1'] + data['SF'] + data['REJ'] + data['S2'] + data['S3'] + data['RSTO'] + data['RSTR'] + data['RSTOS0'] + data['RSTRH'] + data['SH'] + data['SHR'] + data['OTH'] != 1].index)

data = data.drop(data[data['duration'] < 0.0].index)

data = data.drop(data[data['src_bytes'] < 0].index)

data = data.drop(data[data['dst_bytes'] < 0].index)

data = data.drop(data[data['missed_bytes'] < 0].index)

data = data.drop(data[data['src_pkts'] < 0].index)

data = data.drop(data[data['src_ip_bytes'] < 0].index)

data = data.drop(data[data['dst_pkts'] < 0].index)

data = data.drop(data[data['dst_ip_bytes'] < 0].index)

data = data.drop(data[data['tot_bytes'] != data['src_bytes'] + data['dst_bytes']].index)

data = data.drop(data[data['tot_pkts'] != data['src_pkts'] + data['dst_pkts']].index)

data = data.drop(data[~data['label'].isin([0, 1])].index)

data.to_csv(args.generated_data, index=False)