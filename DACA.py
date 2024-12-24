import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import pandas as pd
import re
import duckdb
import numpy as np
from datetime import datetime
import shutil
import argparse

# make joint weightPart
# Objective: Need to test the robustness of the attack method
# STATS :  5-Tables
# IMDB : 5-Tables

def create_bidirectional_mapping(df, primary_key):
    # 确保主键列存在
    if primary_key not in df.columns:
        raise ValueError(f"The primary key '{primary_key}' does not exist in the DataFrame.")
    row_to_key = {}
    key_to_row = {}
    rtol = len(df)
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print('\r', index, '/', rtol, end='', flush=True)
        row_to_key[index] = row[primary_key]
        key_to_row[row[primary_key]] = index
    # 创建行号到主键值的映射
    # row_to_key = {index: row[primary_key] for index, row in df.iterrows()}
    # # 创建主键值到行号的映射
    # key_to_row = {value: index for index, value in row_to_key.items()}

    return row_to_key, key_to_row


def evalDBCard(dbCSVPath='./CorruptDB/ours_stats_simplified_posts.csv_0.2/', dataBase='stats_simplified',
               tableRx='posts.csv', query='stats_CEB.sql', primaryKey="Id"):
    con = duckdb.connect()
    name2pd = {}
    for filename in os.listdir(dbCSVPath):
        if filename.endswith('.csv'):
            file_path = os.path.join(dbCSVPath, filename)
            df = pd.read_csv(file_path)
            con.register(filename.replace('.csv', ''), df)
            name2pd[filename.replace('.csv', '')] = df
    qf = './query/' + query
    qerrList = []
    file = open(qf, 'r')
    lines = file.readlines()
    # print(name2pd.keys())
    shapedTabName = tableRx.replace('.csv', '')
    for idx, line in enumerate(lines):
        print('\rQid:', idx)
        cleaned_line = line.rstrip(';\n')
        print(cleaned_line)
        # 下面的后面需要统一一下，这里有点问题
        sqlQuery = cleaned_line.split("||")[1]
        card = int(cleaned_line.split("||")[0])
        clause = (sqlQuery.split("FROM")[1].split("WHERE")[0]).strip()
        inTab = False
        abrevTab = None
        if tableRx.replace('.csv', '') in clause:
            inTab = True
            # ABBREV
            if dataBase == "stats_simplified":
                abbrev = {'posts': 'p', 'tags': 't', 'users': 'u', 'postHistory': 'ph', 'badges': 'b', 'votes': 'v',
                          'postLinks': 'pl', 'comments': 'c'}
                abrevTab = abbrev[tableRx.replace('.csv', '')]
        if inTab:
            cardNew = con.execute(sqlQuery).df().to_numpy()[0][0]
            qerrList.append(max((cardNew + 1) / (card + 1), (card + 1) / (cardNew + 1)))
            print('Nonzero', len(qerrList), "QERROR:", np.percentile(qerrList, [50, 90, 95, 99, 100]))

            # exit(1)
            # sqlQuery = sqlQuery.replace(" as")


def getJointWeight(dataBase='stats_simplified', tableRx='posts.csv', query='stats_CEB.sql', primaryKey="Id"):
    objectivePath = r'./WeightCache/' + dataBase + tableRx.replace('.', 'D') + query.replace('.', 'D') + primaryKey
    if os.path.exists(objectivePath + '.npy'):
        print(f"文件 '{objectivePath}' 存在。")
        return np.load(objectivePath + '.npy')
    else:
        print(objectivePath)
        print("Cache FileNotFind")
    # print(objectivePath)
    # exit(1)

    # 计算多表连接权重
    # 读取文件并处理每一行
    # 遍历文件夹中的所有文件
    con = duckdb.connect()
    name2pd = {}
    print("Registering")
    t0 = time.time()
    for filename in os.listdir(f"./datasets/" + dataBase + '/'):
        if filename.endswith('.csv'):
            file_path = os.path.join(f"./datasets/" + dataBase + '/', filename)
            df = pd.read_csv(file_path)
            con.register(filename.replace('.csv', ''), df)
            name2pd[filename.replace('.csv', '')] = df
    qf = './query/' + query
    t1 = time.time() - t0
    print("Register takes", t1)
    # sqlT = "SELECT t.id::INTEGER, COUNT(*)  FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2 GROUP BY t.id::INTEGER;"
    # # print('after:', sqlT)
    # VALS = con.execute(sqlT)
    # VALS = VALS.df().to_numpy()
    # r, c = VALS.shape
    # print(VALS)
    #
    # exit(1)
    qerrList = []
    file = open(qf, 'r')
    # with open(qf, 'r') as file:
    lines = file.readlines()
    objTab = name2pd[tableRx.replace('.csv', '')]
    # 构造pk2rid的双射
    print("Making bidirectional_mapping from pk2rid", flush=True)
    t0 = time.time()
    row_to_key, key_to_row = create_bidirectional_mapping(objTab, primaryKey)
    t1 = time.time()
    print("bidirectional_mapping takes:", t1 - t0)
    print(len(row_to_key.keys()))
    print(len(key_to_row.keys()))
    print("Done", flush=True)
    # pk2rid =
    # exit(1)
    rowNumber = int(len(key_to_row.keys()))
    weightColumn = np.zeros((rowNumber, len(lines)))
    print(weightColumn.shape)
    shapedTabName = tableRx.replace('.csv', '')
    for idx, line in enumerate(lines):
        print('\rQid:', idx)
        cleaned_line = line.rstrip(';\n')
        print(cleaned_line)
        # 下面的后面需要统一一下，这里有点问题
        if 'job' in dataBase:
            sqlQuery = cleaned_line
        else:
            sqlQuery = cleaned_line.split("||")[1]
            card = int(cleaned_line.split("||")[0])

        clause = (sqlQuery.split("FROM")[1].split("WHERE")[0]).strip()

        inTab = False
        abrevTab = None
        if tableRx.replace('.csv', '') in clause:
            inTab = True
            # ABBREV
            if dataBase == "stats_simplified":
                abbrev = {'posts': 'p', 'tags': 't', 'users': 'u', 'postHistory': 'ph', 'badges': 'b', 'votes': 'v',
                          'postLinks': 'pl', 'comments': 'c'}
                abrevTab = abbrev[tableRx.replace('.csv', '')]
            elif dataBase == "job":
                abbrev = {'cast_info': 'ci', 'company_type': 'ct', 'movie_companies': 'mc', 'title': 't',
                          'company_name': 'cn', 'keyword': 'k', 'movie_info_idx': 'mi_idx', 'info_type': 'it',
                          'movie_info': 'mi', 'movie_keyword': 'mk'}
                abrevTab = abbrev[tableRx.replace('.csv', '')]
        if inTab:
            if dataBase == "job":
                sqlQuery = sqlQuery.replace('COUNT(*)', abrevTab + '.id::INTEGER, COUNT(*) ').replace(";",
                                                                                                      '') + ' GROUP BY ' + abrevTab + '.id::INTEGER;'
            else:
                sqlQuery = sqlQuery.replace('COUNT(*)', abrevTab + '.Id, COUNT(*) ').replace(";",
                                                                                             '') + ' GROUP BY ' + abrevTab + '.Id;'
            # sqlQuery = sqlQuery.replace(" as")
            print(sqlQuery)
            VALS = con.execute(sqlQuery)
            # print(VALS)
            VALS = VALS.df().to_numpy()
            r, c = VALS.shape
            for i in range(r):
                pkValue = VALS[i, 0]
                rid = key_to_row[pkValue]
                weightColumn[rid, idx] += (VALS[i, 1])
            # print(sum(weightColumn[:,idx]),card)
        else:
            continue
    np.save(objectivePath, weightColumn)
    return weightColumn


def getCard(rij, vector):
    return vector.reshape(1, -1) @ rij


def getCards(rij, vector):
    return vector @ rij


def vecQerr(vecOld, vecNew):
    vecOld = vecOld.reshape(-1)
    vecNew = vecNew.reshape(-1)
    r = vecOld.shape[0]
    qerrList = []
    for i in range(r):
        if vecOld[i] == 0 and vecNew[i] == 0:
            continue
        qerrList.append(
            float(max((vecOld[i].cpu() + 1) / (vecNew[i].cpu() + 1), (vecNew[i].cpu() + 1) / (vecOld[i].cpu() + 1))))
    print('Nonzero', len(qerrList), "Oracle_Qerror:", np.percentile(qerrList, [50, 90, 95, 99, 100]))


def vecQerrcpu(vecOld, vecNew, log=0, f=None):
    vecOld = vecOld.reshape(-1)
    vecNew = vecNew.reshape(-1)
    r = vecOld.shape[0]
    qerrList = []
    for i in range(r):
        if vecOld[i] == 0 and vecNew[i] == 0:
            # qerrList.append(1)
            continue
        qerrList.append(float(max((vecOld[i] + 1) / (vecNew[i] + 1), (vecNew[i] + 1) / (vecOld[i] + 1))))
    print('Queries', len(qerrList), "Oracle_Qerror:", np.percentile(qerrList, [50, 90, 95, 99, 100]), 'Mean:',
          np.mean(qerrList))
    if log == 1:
        QEL = np.percentile(qerrList, [90, 95, 100])
        f.write(str(np.mean(qerrList)) + ',' + str(QEL[0]) + ',' + str(QEL[1]) + ',' + str(QEL[2]) + '\n')


def greedMaxVGPU(rij, ratio):
    m, n = rij.shape
    vector = np.ones(m)
    cardOld = getCard(rij, vector)
    cardEst = cardOld + 0.0
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rij = torch.tensor(rij, dtype=torch.float32).to(device)
    cardOld = torch.tensor(cardOld, dtype=torch.float32).to(device)
    # 将vector转换为PyTorch张量，并移动到GPU上
    vector = torch.ones(rij.shape[0], device='cuda')
    # 初始化cardOld和cardEst为PyTorch张量，并移动到GPU上
    cardOld = getCard(rij, vector)
    cardEst = cardOld.clone().detach()
    # 将gainIfDelete转换为PyTorch张量，并移动到GPU上
    gainIfDelete = (((cardOld + 1) / (cardEst + 1 - rij)) - 1).sum(dim=1)
    gainIfIns = (((cardOld + 1 + rij) / (cardEst + 1)) - 1).sum(dim=1)
    cardEst_last = cardOld.clone().detach()
    vectorsel = torch.ones(m).cuda()
    print(vectorsel.shape)
    rijC = rij.clone().detach() + 0.0
    mask = torch.ones(rij.shape).cuda()
    print('Budget:', int(ratio * rij.shape[0]), rij.shape[0], flush=True)
    for iter in range(int(ratio * rij.shape[0])):
        t0 = time.time()
        max_index_D = gainIfDelete.argmax()
        max_index_I = gainIfIns.argmax()
        if gainIfDelete[max_index_D] > gainIfIns[max_index_I]:
            vector[max_index_D] = 0
            mask[max_index_D, :] = 0
        else:
            vector[max_index_I] = vector[max_index_I] + 1
        cardEst = getCard(rij, vector)
        tx = time.time()
        gainIfDelete = (cardEst / (cardEst + 1 - rij * (mask))).sum(dim=1)
        gainIfIns = ((cardEst + rij * (mask) ) / (cardOld + 1)).sum(dim=1)
        gainIfDelete = gainIfDelete * vector  # 相当于R <- R\v
        gainIfIns = gainIfIns * vector
        t1 = time.time()
        if iter % 100 == 0:
            print(iter, '/', int(ratio * rij.shape[0]))
            print("ETA:", (t1 - t0) * ratio * rij.shape[0] / 60.0, 'min')
            vecQerr(cardOld, cardEst)  # 确保这个函数也可以处理PyTorch张量
    return vector.cpu().numpy()  # 如果需要返回NumPy数组，将张量移回CPU


class Attacker:
    def __init__(self, ratio=0.2, dataBase='stats_simplified', tableRx='posts.csv', query='stats_CEB.sql',
                 primaryKey="Id"):
        """
        攻击者集成
        :param ratio: 破坏的比例
        :param dataBase: 数据库
        :param tableRx: 被破坏的表
        :param query: 给定的查询负载
        :param primaryKey: 主键 STATS:Id     IMDB:id  注意大小写
        """
        self.ratio = ratio
        self.dataBase = dataBase
        self.tableRx = tableRx
        self.query = query
        self.primaryKey = primaryKey
        self.DB_TABLE_QUERY_pk_Name = dataBase + tableRx.replace('.', 'D') + query.replace('.', 'D') + primaryKey
        self.rij = getJointWeight(dataBase, tableRx, query, primaryKey)

    def genCorruptDB(self, method):
        atkVec = None
        # 集成多种方法
        if method == "ours":
            path = r'./WeightCache/Ours_' + str(self.ratio) + self.DB_TABLE_QUERY_pk_Name
            if os.path.exists(path + '.npy'):
                # 首先检查是否算过了攻击向量
                atkVec = np.load(path + '.npy').reshape(1, -1)
                cardOld = np.ones((1, atkVec.shape[1])) @ self.rij
                cardNew = atkVec @ self.rij
                # vecQerrcpu(cardOld, cardNew)
                vecQerrcpu(cardOld, cardNew)
                # 如果没有算过，重新生成
                return
            else:
                t0 = time.time()
                print("Generation attack vector")
                atkVec = greedMaxVGPU(self.rij, self.ratio).reshape(1, -1)
                t1 = time.time()
                print("Takes:", t1 - t0, '(s)', "Ratio:", self.ratio)
                print("Done")
                print("Saving cache")
                atkVec = np.array(atkVec)
                np.save(path + '.npy', atkVec)
                cardOld = np.ones((1, atkVec.shape[1])) @ self.rij
                cardNew = atkVec @ self.rij
                print("FinalCheck:", self.ratio, self.tableRx)
                vecQerrcpu(cardOld, cardNew)

        elif method == "clean":
            atkVec = np.ones((1, self.rij.shape[0]))
            cardOld = np.ones((1, atkVec.shape[1])) @ self.rij
            cardNew = atkVec @ self.rij
            vecQerrcpu(cardOld, cardNew)


        elif method == "random":
            atkVec = np.ones((1, self.rij.shape[0]))
            # 计算总共有多少个元素
            total_elements = self.rij.shape[0]
            # 计算20%的元素数量
            num_zeros = int(total_elements * self.ratio)
            # 随机生成num_zeros个索引
            indices_to_zero = np.random.choice(total_elements, num_zeros, replace=False)
            # 将这些索引对应的元素置为零
            # 因为atkVec是一个二维数组，我们需要将其展平为一维数组进行操作
            atkVec = atkVec.reshape(-1)
            atkVec[indices_to_zero] = 0
            atkVec = atkVec.reshape(1, -1)
            cardOld = np.ones((1, atkVec.shape[1])) @ self.rij
            cardNew = atkVec @ self.rij
            vecQerrcpu(cardOld, cardNew)

        # 输出数据库
        out_Path = './CorruptDB/' + method + '_' + self.dataBase + '_' + self.tableRx + "_" + str(self.ratio) + '/'
        srcDB_Path = f"./datasets/" + self.dataBase + '/'
        if not os.path.exists(out_Path):
            os.mkdir(out_Path)
            print("OutPut Dir Made,rearranging data")
        else:
            print("OutPut Dir Exists,rearranging data")
        for filename in os.listdir(srcDB_Path):
            if filename.endswith('.csv'):
                table = filename.replace('.csv', '')
                print(filename, self.tableRx)
                if filename == self.tableRx:
                    print("Remakine Rx")
                    f_write = open(out_Path + filename, 'w')
                    f_read = open(srcDB_Path + filename, 'r')
                    lines = f_read.readlines()
                    f_write.write(lines[0])
                    lines = lines[1:]
                    print("Before:", len(lines))
                    writeNum = 0
                    for idx, li in enumerate(lines):
                        if atkVec[0, idx] == 1:
                            f_write.write(li)
                            writeNum += 1
                        else:
                            continue
                    print("After:", writeNum)
                else:
                    shutil.copy(srcDB_Path + filename, out_Path + filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='stats_simplified', help='Which dataset to be poisoned')
    parser.add_argument('--table', default='posts.csv', help='Which table to be poisoned')
    parser.add_argument('--budget', default=0.2, help='Attack budget to the current table')
    parser.add_argument('--primaryKey', default="Id", help='Attacked table\'s  PK')
    parser.add_argument('--query', default="stats_CEB.sql", help='Testing query')
    parser.add_argument('--method', default="ours", help='ATK Method')
    
    args = parser.parse_args()
    print("="*20,'Parameters Confirming',"="*20)
    print(" ratio:",args.budget,'\n', "dataBase:",args.dataset,'\n', "tableRx:",args.table,'\n', "query:",args.query,'\n',
                           "primaryKey:",args.primaryKey)
    print("="*50)
    ATK = Attacker(ratio=args.budget, dataBase=args.dataset, tableRx=args.table, query=args.query,
                           primaryKey=args.primaryKey)
    ATK.genCorruptDB(args.method)

    