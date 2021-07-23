#-*-coding:utf-8-*-
import sys
import os
import numpy as np
import random
from copy import deepcopy
project = str(sys.argv[1]) + "/"

def tqdm(a):
    try:
        from tqdm import tqdm
        return tqdm(a)
    except:
        return a

vocabulary = {}
vocabulary["<Start>"] = 2
vocabulary["NothingHere"] = 0
vocabulary["Unknown"] = 1
char_vocabulary = {}
char_vocabulary["default"] = 0
char_vocabulary["Unknown"] = 1
tree_vocabulary = {}
tree_vocabulary["Unknown"] = 1
tree_vocabulary["NothingHere"] = 0
tree_vocabulary["NoneCopy"] = 2
tree_vocabulary["CopyNode"] = 3
tree_vocabulary["End"] = 5
tree_vocabulary["<StartNode>"] = 4
Rule = []
Nonterminal = []

copy_ast = ""

trainset = []       # for a dataset, 
trainset.append([]) # train for trainset[0]
trainset.append([]) # dev for trainset[1]
trainset.append([]) # test for trainset[2]

is_train = False


nl_len = 200
tree_len = 500
parent_len = 20
rules_len = rulelist_len = 200
char_len = 10
function_len = 1
nl_voc_ground = {}

length = [nl_len, tree_len, tree_len, tree_len, parent_len, rulelist_len, rulelist_len, function_len, rulelist_len]

dev_count = 0
dev_ab = 0

def readrules():
    f = open(project + "Rule.txt", "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        l = []
        words = line.split()
        l.append(words[0])
        Nonterminal.append(words[0])
        l1 = []
        for i in range(2, len(words)):
            l1.append(words[i])
        l.append(l1)
        Rule.append(l)

readrules()
rulesnum = len(Rule)
classnum = rulesnum + nl_len

def readvoc ():
    global comment_vocabulary
    global tree_vocabulary
    global char_vocabulary
    global vocabulary
    f = open(project + "tree_voc.txt", "r")
    '''lines = f.readlines()
    for line in lines:
        words = line.strip().split()
        if int(words[1]) >= 10:
            tree_vocabulary[words[0]] = len(tree_vocabulary)'''
    tree_vocabulary = eval(f.readline())
    if "HS" not in project:
        tree_vocabulary["End"] = len(tree_vocabulary)
    f.close()

    f = open(project + "char_voc.txt", "r")
    lines = f.readlines()
    try:
        char_vocabulary = eval(lines[0])
    except:
        #lines = f.readlines()
        for line in lines:
            words = line.strip().split()
            if int(words[1]) >= 10:
                char_vocabulary[words[0]] = len(char_vocabulary)
    f.close()

    f = open(project + "nl_voc.txt", "r")
    '''lines = f.readlines()
    for line in lines:
        words = line.strip().split()
        if int(words[1]) >= 2:
            vocabulary[words[0]] = len(vocabulary)
        nl_voc_ground[words[0]] = len(nl_voc_ground)'''
    vocabulary = eval(f.readline())
    f.close()
    

def word2vec (word, tp):
    is_train = False
    if tp == "nl":
        if word not in vocabulary:
            #if not is_train:
#            print (copy_ast)
            return 1
            #comment_vocabulary[word] = len(comment_vocabulary)
        return vocabulary[word]
    else:
        if word not in tree_vocabulary:
            #if not is_train:
            return 1
            #tree_vocabulary[word] = len(tree_vocabulary)
        return tree_vocabulary[word]

def line2vec (line, tp, length):
    vec = np.zeros([length])
    tokens = line.split()
    if "tree" == tp:
        for i in range(min(len(tokens), length)):
            vec[i] = word2vec(tokens[i], "tree")
    else:
        for i in range(min(len(tokens), length)):
            vec[i] = word2vec(tokens[i], "nl")
        
    return vec

def rule2classnum (line):
    line = str(line)
    numbers = line.strip().split()
    l = []
    #print (line)
    for n in numbers:
        num = int(n)
        if num == 9999:
            print (num)
            exit()
        if num >= 10000: # copy
            num = num - 10000 + rulesnum
        if num >= classnum: # check length
#            print (num)
#            print ("NL Length is not enough")
            l.append(classnum + 2) 
        else:
            l.append(num)

    return l

def rule2classvec(line, length):
    l = rule2classnum(line)
    ret = np.zeros([length])
    for i in range(min(length, len(l))):
        ret[i] = l[i]
    return ret

def rulebondast (num, father, nl):
    if num == 0:
        vec = np.zeros([1])
        vecson = np.zeros([10])
        #for i in range(len(vec)):
        #    vec[i] = 1
        #for i in range(len(vecson)):
        #    vecson [i] = 1
        return vec, vecson
    #print (num)
    rule = rule2classnum(num - 1)[0]
    #print (rule)
    vec = np.zeros([1])
    vecson = np.zeros([10])
    #for i in range(len(vecson)):
    #    vecson[i] = 1 # NothingHere
    words = nl.split()
    #print (words)
    if rule >= rulesnum and rule < classnum:
        #print (rule)
        site = rule - rulesnum
        #vec[0] = word2vec(father, "tree")
        vec[0] = father#word2vec("CopyNode", "tree")
        #print (words)
        #print (words[site])
        if site >= len(words):
            #print (site)
            exit()
            vecson[0] = word2vec("NoneCopy", "tree")
        else:
           # print (words[site])
            vecson[0] = word2vec(words[site] + "", "tree")
    elif rule >= classnum:
        vec[0] = word2vec("<StartNode>", "tree")
        vecson[0] = word2vec("root", "tree")
    else:
        vec[0] = word2vec(Rule[rule][0], "tree")
        l = Rule[rule][1]
        for i in range(min(10, len(l))):
            vecson[i] = word2vec(l[i], "tree")
    return vec, vecson

def line2rules (line, length, father, nl):
    vec = np.zeros([length])
    vecson = np.zeros([length, 10])
    words = line.split()
    l = []
    l.append(classnum + 2) # <Start Node>
    for word in words:
        num = int(word)
        l.append(num)
    #while len(l) < length:
    #    l.append(0)
    #print (len(l))
    #print (len(father))
    for i in range(min(length, len(l))):
        #if l[i] == classnum + 2:
        #    continue
            #vec[i] = 0#word2vec("NothingHere", "tree")
        #elif i >= len(father):
        #    v1, v2 = rulebondast(l[i], "", nl)
        #    vec[i] = v1[0]
        #    for t in range(10):
        #        vecson[i][t] = v2[t]
        #else:
        v1, v2 = rulebondast(l[i], father[i - 1], nl)
        vec[i] = v1[0]
        for t in range(10):
            vecson[i][t] = v2[t]
    return vec, vecson

def one_hot(num, length):
    onehot = np.zeros([length])
    if num >= length: # check length
        print ("NL Length is not enough one-hot")
        return onehot 
    onehot[num] = 1
    return onehot

def char2num (c):
    global char_vocabulary
    if c not in char_vocabulary:
        return 1
        #char_vocabulary[c] = len(char_vocabulary)
    return char_vocabulary[c]

def line2charvec (line, length, charlength):
    vec = np.zeros([length, charlength])
    tokens = line.strip().split()
    for i in range(min(length, len(tokens))):
        for t in range(min(charlength, len(tokens[i]))):
            vec[i, t] = char2num(tokens[i][t])
    return vec

def line2rulevec (line, length):
    vec = np.zeros([length])
    tokens = line.strip().split()
    vec[0] = classnum + 2
    for i in range(min(length -1, len(tokens))):
        vec[i + 1] = rule2classnum(int(tokens[i]) - 1)[0] + 1
    return vec

def line2ground (line, length):
    vec = np.zeros([length])
    words = line.strip().split()
    for i in range(len(words)):
        if words[i] not in nl_voc_ground:
            continue
        vec[i] = nl_voc_ground[words[i]]
    return vec

def read_tree_path(tree_path, rules_line, nowsite):
    all_lines = len(rules_line.strip().split())
    fathers = []
    tree_path_len = 10
    tree_path_vec = np.zeros([length[5], tree_path_len])
    for i in range(nowsite, nowsite + all_lines):
        t = i - nowsite
        if t >= length[5]:
            break
        line = tree_path[i]
        words = line.strip().split()
        for w_site in range(min(len(words), tree_path_len)):
            tree_path_vec[t][w_site] = word2vec(words[w_site], "tree")
        fathers.append(word2vec(words[0], "tree"))
    
    return tree_path_vec, fathers

def get_father_list(nodes):
    father_list = [-1]
    stack = []
    nowsite = nodes[0]
    stack.append([nowsite, 0])
    for i in range(1, len(nodes)):
        while True:
            top = stack[-1]
            if nodes[i] > top[0]:
                father_list.append(top[1])
                stack.append([nodes[i], i])
                break
            else:
                stack.pop()
    return father_list

def build_lca(nodes_deepth):
    node = nodes_deepth
    fathers = []
    father_list = nodes_deepth#get_father_list(nodes_deepth)
    used = []
    for i in range(len(node)):
        par = []
        index = i
        while index != -1:
            par.append(index)
            index = father_list[index]
            if index not in used:
                used.append(index)
        fathers.append(par[::-1])
    term = []
    for i in range(len(node)):
        if i not in used:
            term.append(i)

    return fathers, term


def query_lca(start, end, fathers, nodes_deepth):
    try:
        l_s = len(fathers[start])
        l_e = len(fathers[end])
    except:
        return 1000000
    same_site = 0
    for i in range(min(l_s, l_e)):
        if fathers[start][i] == fathers[end][i]:
            same_site = i
        else:
            break
    return len(fathers[start]) + len(fathers[end]) - 2 * (1 + same_site)

    
def line2mask(lines, length):
    nodes_deepth = lines.strip().split()
    nodes_deepth = [int(-1)] +  nodes_deepth
    for i in range(1, len(nodes_deepth)):
        nodes_deepth[i] = int(nodes_deepth[i]) + 1

    ret = np.zeros([length, length])
    fathers, term = build_lca(nodes_deepth)
    labels = np.zeros([length])

    father_vec = np.zeros([length, length])
    termnow = -1
    for i in range(length):
        try:
            site = fathers[i][-1] # next node
        except:
            break
        labels[i] = len(fathers[i])
        try:
            ret[i][fathers[i][-2]] = 1.0
        except:
            ret[i][fathers[i][-1]] = 1.0
    return ret, father_vec, labels
 

def read_data (file_name):
    file2number = {}
    file2number["train_trans.txt"] = 0
    file2number["dev_trans.txt"] = 1
    file2number["test_trans.txt"] = 2

    index_of_dataset = file2number[file_name]
    f = open(project + file_name, "r")
    file_data = []
    for i in range(8):
        file_data.append([])

    number2file = {}
    number2file[0] = project + "train_tree.txt"
    number2file[1] = project + "dev_tree.txt"
    number2file[2] = project + "test_tree.txt"
    f_tree = open(number2file[index_of_dataset], "r")
    tree_path = f_tree.readlines()
    f_tree.close()

    lines = f.readlines()
    all_vec = []
    count = 0 
    each_vec = []

    bf = ""
    rules_line = ""
    father = []
    now_site = 0
    for i in tqdm(range(len(lines))):
        lines[i] = str(lines[i]).strip()
        t = i % 9
        if t == 0 : # the first line; Natural Languages;
            if bf != "":
                all_vec.append(deepcopy(each_vec))
            
            if bf != lines[i] and bf != "" and len(bf.split()) < length[0]: # length protection
                trainset[index_of_dataset].append(deepcopy(all_vec))
                all_vec = []
                father = []
                # 50% datas are selected
                #if index_of_dataset in [0, 2, 1] and i >= len(lines) // 20:
                #    return


            bf = lines[i]

            if len(lines[i].split()) >= length[0]:
                all_vec = []
            each_vec = []
            # Using vocabulary
            each_vec.append(line2vec(lines[i], "nl", length[t]))
            # Using char_vocabulary for char embeddding; 
            each_vec.append(line2charvec(lines[i], length[t], char_len))

        elif t == 6: # the line denotes the target output 
            each_vec.append(rule2classvec(lines[i], length[t]))
        elif t == 5: # the line denotes the predict rules
            each_vec.append(line2rulevec(lines[i], length[t]))
            rules_line = lines[i]
        elif t == 8:
            vv, vvv, labels = line2mask(lines[i].strip(), length[t])
            each_vec.append(vv)
            each_vec.append(vvv)
            each_vec.append(labels)
            #each_vec.append(line2mask(lines[i].strip(), length[t]))
        else: 
            each_vec.append(line2vec(lines[i], "tree",length[t]))
        
        if t == 7:
            each_vec.append(bf)
            tp_vec, fathers_vec = read_tree_path(tree_path, rules_line, now_site)
            each_vec.append(tp_vec)
            #print (each_vec[-1])
            #exit()
            now_site += len(rules_line.strip().split())
            bf = "<Start> " + bf
            each_vec.append(line2vec(bf, "nl", length[0]))
            each_vec.append(line2charvec(bf, length[0], char_len))
            each_vec.append(line2ground(bf.replace("<Start> ", ""),length[0]))
            #print (now_site)

        if t == 5:
            tp_vec, fathers_vec = read_tree_path(tree_path, rules_line, now_site)
            #print (len(tp_vec) == len(fathers_vec))
            v1, v2 = line2rules(lines[i], length[t], fathers_vec, bf)
            each_vec.append(v1)
            each_vec.append(v2)
        if t == 4:
            father.append(lines[i].split()[-1].replace("_root", ""))



    # the last data; 
    all_vec.append(deepcopy(each_vec))
    trainset[index_of_dataset].append(deepcopy(all_vec))



def random_data (dataset): # shuffle training set  
    #num_lst = np.random.permutation(range(int(dataset.shape[0])))
    random.shuffle(dataset)

def batch_data (batch_size, dataset_name): # get an acceptable data for NN;
    dic = {}
    dic["train"] = 0
    dic["dev"] = 1
    dic["test"] = 2
    index_of_dataset = dic[dataset_name]

    global trainset
    data = trainset[index_of_dataset]
    #print (data)
    all_data = []
    all_index = []
    data_now = []

    for i in range(len(data)):
        data_now += [[data[i][q][t] for t in range(19)] for q in range(len(data[i]))]

    if index_of_dataset == 0: # random training data;
        random.shuffle(data_now)
    
    #data_now = np.array(data_now)
    all_data = [data_now[site: min(len(data_now), site + batch_size)] for site in range(0, len(data_now), batch_size)]
    #print (all_data[0][0])
    ret_data = []
    for site in range(len(all_data)):
        ret_data += [[[all_data[site][i][t] for i in range(len(all_data[site]))] for t in range(19)]]
    #print (all_data[0][0])
    #print (len(all_data))
    #print (len(all_data[0]))
    return ret_data, all_index

    if index_of_dataset == 0: # random training data;
        random.shuffle(data)

    all_data = []
    all_index = [] # denotes the card number;
                   # the 1st dim denotes the batch number;
                   # the 2nd dim denotes the 9-grams;
                   # the 3rd dim denotes the data (e.g., NL, Char, ...)


    batch_data = [] # denotes the each batches of data;
                    # the 1st dim denotes the 9-grams;
                    # the 2nd dim denotes the data (e.g., NL, Char, ...)
    for t in range(19):
        batch_data.append([])
    batch_index = []

    index_num = 0
    for i in range(len(data)): # i denotes the index of the card;
        if len(batch_data[0]) >= batch_size:
            for t in range(len(batch_data) - 1): # the batch_data[-1]: natural language
                batch_data[t] = np.array(batch_data[t])
            all_data.append(deepcopy(batch_data))
            all_index.append(deepcopy(batch_index))
            batch_data = []
            for t in range(19):
                batch_data.append([])
            batch_index = []
            index_num = 0

        # for a data:
            # the 1st dim means: the cards
            # the 2nd dim means: the number of tuples.
            # the 3rd dim denotes each of the 9-gram tuples (e.g., NL, Char).
        for t in range(len(batch_data)):
            batch_data[t].extend([data[i][q][t] for q in range(len(data[i]))])
        print (batch_data[9])
        print (batch_data[7])
        exit()
        batch_index.extend([index_num] * len(data[i]))
        index_num += 1

    if (len(batch_data[0]) > 0):
        # the last batch; 
        
        for t in range(len(batch_data) - 1):
            batch_data[t] = np.array(batch_data[t])
        all_data.append(deepcopy(batch_data))
        all_index.append(deepcopy(batch_index))
    
    return all_data, all_index

def resolve_data():
    global trainset
    readvoc()
    read_data("train_trans.txt")
    read_data("dev_trans.txt")
    read_data("test_trans.txt")

if sys.argv[0] == "run.py":
    resolve_data()
elif "predict" in sys.argv[0]:
    readvoc()
