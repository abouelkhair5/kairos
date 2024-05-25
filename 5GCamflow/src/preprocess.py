# Input: StreamSpot dataset
# Output: Vectorized graphs

import functools
import os
import json
import re
import torch
from tqdm import tqdm
from torch_geometric.data import *

# If the datum have already been loaded in the database before, set this as False.
# Set it as True if it is the first time running this code.
process_raw_data = True


import psycopg2
from psycopg2 import extras as ex

connect = psycopg2.connect(database = '5gcamflow',
                           host = '/var/run/postgresql/',
                           user = 'postgres',
                           password = 'postgres',
                           port = '5432'
                           )

# Create a cursor to operate the database
cur = connect.cursor()
# Rollback when there exists any problem
connect.rollback()


node_types = set()
edge_types = set()
if process_raw_data:
    fidx = 0
    path = "/the/absolute/path/of/raw_log"  # The paths to the dataset.
    datalist = []
    with open(path) as f:
        for line in tqdm(f):
            src_id, dest_id, types = line.strip('\n').split(' ')
            src_type, dest_type, edge_type, ts = types.split(":")
            spl = [
                src_id,
                src_type,
                dest_id,
                dest_type,
                edge_type,
                ts,
                fidx,
                ]
            datalist.append(spl)
            node_types.add(src_type)
            node_types.add(dest_type)
            edge_types.add(edge_type)
            if len(datalist) >= 10000:
                sql = '''insert into raw_data
                 values %s
                '''
                ex.execute_values(cur, sql, datalist, page_size=10000)
                connect.commit()
                datalist = []

nodevec = torch.nn.functional.one_hot(torch.arange(0, len(node_types)), num_classes=len(node_types))
edgevec = torch.nn.functional.one_hot(torch.arange(0, len(edge_types)), num_classes=len(edge_types))

edge2onehot = {}
node2onehot = {}
c = 0
for i in node_types:
    node2onehot[i] = nodevec[c]
    c += 1
c = 0
for i in edge_type:
    edge2onehot[i] = edgevec[c]
    c += 1

os.system("mkdir -p ../data/")
for graph_id in tqdm(range(600)):
    sql = "select * from raw_data where graph_id='{graph_id}' ORDER BY _id;".format(graph_id=graph_id)
    cur.execute(sql)
    rows = cur.fetchall()
    from torch_geometric.data import TemporalData

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for i in rows:
        src.append(int(i[0]))
        dst.append(int(i[2]))
        msg_t = torch.cat([node2onehot[i[1]], edge2onehot[i[4]], node2onehot[i[3]]], dim=0)
        msg.append(msg_t)
        t.append(int(i[-1]))    # Use logical order of the event to represent the time

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, "../data/graph_" + str(graph_id) + ".TemporalData")

print("end")
