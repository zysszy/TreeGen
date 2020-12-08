#-*-coding:utf-8-*-
import sys
from code_generate_model import *
from resolve_data import *
import os
import tensorflow as tf
import numpy as np
import os
import math
import queue as Q
from copy import deepcopy
from tqdm import tqdm
project = str(sys.argv[1]) + "/"

os.environ["CUDA_VISIBLE_DEVICES"]="5"

vocabu = {}
tree_vocabu = {}
vocabu_func = {}
tree_vocabu_func = {}
vocabu_var = {}
tree_vocabu_var = {}

J_HasSon = []
J_VarList = []
J_readrulenum = -1
J_NeedsEnd = []
J_NlList = []


global_step = 0
embedding_size = 128
conv_layernum = 128
conv_layersize = 3
rnn_layernum = 50
batch_size = 64
NL_vocabu_size = len(vocabulary)
Tree_vocabu_size = len(tree_vocabulary)
NL_len = nl_len
Tree_len = tree_len
learning_rate = 1e-5
keep_prob = 0.5
pretrain_times = 0
pretrain_dis_times = 2
train_times = 1000
parent_len = 20

rulelist_len = 200

step_list_p = []
numberstack = []
list2wordlist = []
copy_node = []
cardnum = []


def J_readrule():
    f = open(project + "Rule.txt", "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        J_HasSon.append(line.strip().split()[0])
    f = open(project + "copy_node.txt", "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        J_HasSon.append(line.strip().split()[0])
    f = open(project + "WithListEnd.txt", "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        J_HasSon.append(line.strip().split()[0])
        J_NeedsEnd.append(line.strip().split()[0])

J_readrule()

def J_readlist(in_file):
    lines = in_file.readlines()
    global J_readrulenum
    J_readrulenum = int(lines[1])
    return lines[0].replace(" node_gen ^", "").strip().split()

def J_findtheson_name(l, site):
    ans = []
    count = 0
    nowsite = site
    while (nowsite < len(l)):
        if l[nowsite] == "^":
            count -= 1
        else:
            if count == 1:
                ans.append(l[nowsite])
            count += 1
        if count <= 0:
            break
        nowsite += 1
    return ans

def J_isend(l, site):
    if l[site] not in J_HasSon:
        return True
    if l[site] == "End" or l[site] == "^":
        return True
    sonlist = J_findtheson_name(l, site)
    if len(sonlist) == 0:
        return False
    elif l[site] not in J_NeedsEnd:
        return True
    elif l[site] in J_NeedsEnd and sonlist[-1] == "End":
        return True
    return False

def J_findthecontrol(liststr, site):
    ans = [site + 1]
    count = 0
    nowsite = site
    while (nowsite < len(liststr)):
        if liststr[nowsite] == "^":
            count -= 1
        else:
            count += 1
        if count <= 0:
            ans.append(nowsite)
            break
        nowsite += 1
    return ans

def J_AddOneSon(l, rule, site):
    node = l[site]
    if node != rule[0]:
        assert False
    se = J_findthecontrol(l, site)
    e = se[1]
    s = se[0]
    newlist = deepcopy(l)
    newlist.insert(e, "^")
    step_list_p.insert(e, global_step)
    newlist.insert(e, rule[1][0])
    step_list_p.insert(e, global_step)
    return newlist

def J_AddSon(l, rulenum, site):
    if rulenum >= len(Rule): # for copy
        newlist = deepcopy(l)
        newlist.insert(site + 1, "^")
        step_list_p.insert(site + 1, global_step)
        newlist.insert(site + 1, J_NlList[rulenum - len(Rule)])
        step_list_p.insert(site + 1, global_step)
        return newlist

    newlist = deepcopy(l)
    rule = Rule[rulenum]
    for son in rule[1][::-1]:
        newlist.insert(site + 1, "^")
        step_list_p.insert(site + 1, global_step)
        newlist.insert(site + 1, son)
        step_list_p.insert(site + 1, global_step)
    return newlist

def J_AddSon_nodegen(l, site):
    newlist = deepcopy(l)
    se = J_findthecontrol(l, site)
    newlist.insert(se[1], "^")
    #step_list_p.insert(se[1], global_step)
    newlist.insert(se[1], "node_gen")
    #step_list_p.insert(se[1], global_step)
    return newlist

father_index_now = -1

def J_scan(l, rulenum):
    for i in range(len(l)):
        now = l[i]
        if now == "^":
            continue
        if not J_isend(l, i):
            global father_index_now
            father_index_now = step_list_p[i]
            if l[i] in J_NeedsEnd:
                return J_AddOneSon(l, Rule[J_readrulenum], i)

            return J_AddSon(l, J_readrulenum, i)
    return None

def J_findthefather_site(l, site):
    index = site - 1
    count = 0
    if index < 0:
        return -1
    while index >= 0:
        #print ("fa", words[i])
        if "^" in l[index]:
            count -= 1
        else:
            count += 1
        if count == 1:
#            exit()
#            print (words[index])
            #return words[index]
            return index
        index -= 1
    return -1

def J_scan_for_node(l, rulenum):
    for i in range(len(l)):
        now = l[i]
        if now == "^":
            continue
        if not J_isend(l, i):
            newl = J_AddSon_nodegen(l, i)

            return newl, i#J_findthefather_site(newl, i + 1)
    return None, -1

def J_getfeaturenode(l, nextsite):
    i = nextsite
    node_par = []
    node_par.append(l[i])
    par = J_findthefather_site(l, i)
    pars = ""
    if par == -1:
        pars = "Unknown"
    else:
        pars = l[par]
    node_par.append(pars)
    l_f = []
    par = i
    while (par != -1):
        l_f.append(l[par])
        par = J_findthefather_site(l, par)
    node_par.append(" ".join(l_f))
    return node_par

def J_run():
    global global_step
    in_file = open(project + "Tree_Rule.in")
    #in_file.close()
    fw = open(project + "Tree_Feature.out", "w")
    l = J_readlist(in_file)
    in_file.close()
    global_step += 1
    newl = J_scan(l, J_readrulenum)
    if newl == None:
        fw.write(" ".join(l) + "\n")
        fw.write("END\n")
    else:
        newl1, nextsite = J_scan_for_node(newl, J_readrulenum)
        if newl1 == None:
            fw.write(" ".join(newl) + "\n")
            fw.write("END\n")
        else:
            newl = newl1
            node_par = J_getfeaturenode(newl, nextsite)
            out = " ".join(newl)
            fw.write(out.replace(" node_gen ^", "") + "\n")
            fw.write(node_par[0] + "\n")
            fw.write(node_par[1] + "\n")
            fw.write(node_par[2] + "\n")
            fw.write(out.replace(" End ^", "") + "\n")
            fw.write(out.replace(" End ^", "") + "\n")
            fw.write("1\n")
            fw.write("1\n")
            fw.write(out.replace(" End ^", "") + "\n")
            fw.write("1\n")
            fw.write(str(father_index_now) + "\n")
    fw.close()

def create_model(session, g, placeholder=""):
    if(os.path.exists(project + "save1")):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(project + "save1/"))
        print("load the model")
    else:
        classvec = data_random()
        session.run(tf.global_variables_initializer(), feed_dict={d.variable:classvec})
        print("create a new model")


class Javaoutput:
    def __init__(self, Tree, Nl, Node, PNode , Root, TreeWithEnd,FatherTree, GrandFatherTree, state):
        self.Tree = Tree
        self.Nl = Nl
        self.Node = Node
        self.PNode = PNode
        self.Root = [Root]
        self.Probility = 1
        self.is_end = False
        self.state = state
        self.FuncDict = {}
        self.FuncList = []
        self.VarList = []
        self.father_index = []
        self.rule = []
        self.RuleList = []
        self.DeepthList = []
        self.FatherTree = FatherTree
        self.TreeWithEnd = TreeWithEnd
        self.GrandFatherTree = GrandFatherTree
        self.list2wordlistjava = []
        self.numberstack = []
        self.step_list = [-1] * 30
        self.gs = -1

    def prin(self):
        print(self.Tree)

    def __lt__(self, other):
        return self.Probility > other.Probility

def getJavaOut(Nl):
    f = open(project + "Tree_Feature.out", "r")
    lines = f.readlines()
    f.close()
    # print(lines)
    if len(lines) == 2:
        return Javaoutput(lines[0][:-1], Nl, "", "", "", "", "", "", "end")
    if len(lines) == 12:
        return Javaoutput(lines[4][:-1], Nl, lines[1][:-1], lines[2][:-1], lines[3][:-1], lines[0][:-1],lines[6][:-1], lines[7][:-1], "end")

    if len(lines) == 1:
        return Javaoutput("", Nl, "", "", "", "", "", "", "error")
    return Javaoutput(lines[4][:-1], Nl, lines[1][:-1], lines[2][:-1], lines[3][:-1], lines[0][:-1],lines[6][:-1], lines[7][:-1], "grow")

def getlistDeep_all(inputlist):
    ne = []
    count = 0
    for p in inputlist:
        if p == "^":
            count -= 1
            ne.append(count)
        else:
            ne.append(count)
            count += 1
    return ne

def cov(tree):
    ans = " "
    li = tree.split()
    #for s in str:
    deeplist = getlistDeep_all(li)
    mp = {}
    for i in range(len(li)):
        if li[i] == "^":
            now = deeplist[i]
            li[i] = mp[now] + "^"
        else:
            mp[deeplist[i]] = li[i]
        ans += " " + li[i]
    return ans.replace("  ", "")

def pre_mask():
    mask = np.zeros([rulelist_len, rulelist_len])
    for i in range(rulelist_len):
        for t in range(i + 1):
            mask[i][t] = 1
    return mask

def g_predict_beam(sess, model, batch_data):
    batch = batch_data
    rewards = np.zeros([len(batch[1])])

    for i in range(len(rewards)):
        rewards[i] = 1

    y = sess.run(model.y_result, feed_dict={model.input_NL: batch[0],
                                                model.input_NLChar:batch[1],
                                                model.inputrulelist:batch[6],
                                                model.inputrulelistnode:batch[7],
                                                model.inputrulelistson:batch[8],
                                                model.tree_path_vec: batch[9],
                                                model.treemask: batch[10],
                                                model.father_mat: batch[11],
                                                model.labels:batch[12],
                                                model.antimask:pre_mask(),
                                                model.keep_prob: 1,
                                                model.rewards: rewards,
                                                model.is_train: False
                                                })

    for i in range(len(batch[6][0])):
        if batch[6][0][i] == 0:
            return y[0][i - 1]
    
    return y[ -1 ]

def get_tree_path_vec_for_pre (tree_path):
    fathers = []
    tree_path_len = 10
    tree_path_vec = np.zeros([length[5], tree_path_len])
    #return tree_path_vec
    for i in range(len(tree_path)):
        words = tree_path[i].strip().split()
        for t in range(min(len(words), tree_path_len)):
            tree_path_vec[i][t] = word2vec(words[t], "tree")
        fathers.append(word2vec(words[0], "tree"))
    return tree_path_vec, fathers


step = 1
def getAction(sess, Code_gen_model, JavaOut):
    valid_batch, _ = batch_data(1, "test") # read data 
    input_NL = line2vec(JavaOut.Nl, "nl", length[0])
    input_NLChar = line2charvec(JavaOut.Nl, length[0], char_len)
    input_Tree = line2vec(cov(JavaOut.Tree), "tree", length[1])
    input_Father = line2vec(cov(JavaOut.FatherTree), "tree", length[2])
    input_Grand = line2vec(cov(JavaOut.GrandFatherTree), "tree", length[3])
    tree_path_vec, father_vec = get_tree_path_vec_for_pre(JavaOut.Root)
    print (JavaOut.Root)
    deepthlist = []
    tree_path = JavaOut.Root
    for i in range(len(tree_path)):
        words = tree_path[i].strip().split()
        deepthlist.append(str(len(words)))


    root = ""
    rules_str = ""
    rules_destart = ""
    flag = True
    for n in JavaOut.RuleList:
        rules_str += str(n) + " "
    
    input_Rules = line2rulevec(rules_str, length[5])
    input_func = np.zeros([1])
    list_input = []
    list_input.append(input_NL)
    list_input.append(input_NLChar)
    list_input.append(input_Tree)
    list_input.append(input_Father)
    list_input.append(input_Grand)
    list_input.append("")
    list_input.append(input_Rules)
    v1, v2 = line2rules(rules_str, length[5], father_vec, JavaOut.Nl)
    list_input.append(v1)
    list_input.append(v2)
    global step
    step += 1
    list_input.append(tree_path_vec)
    deepth = " ".join(deepthlist)
    print ("------")
    print (JavaOut.father_index)
    line = ""
    for f in JavaOut.father_index:
        line += str(f) + " "
    print (line)
    ret, father_vec, labels = line2mask(line, length[5])
    list_input.append(ret)
    list_input.append(father_vec)
    list_input.append(labels)
    for i in range(len(list_input)):
        list_input[i] = np.expand_dims(list_input[i], axis=0)
    
    return g_predict_beam(sess, Code_gen_model, list_input)

def WriteJavaIn(JavaOut, action):
    f = open(project + "Tree_Rule.in", "w")
    f.write(JavaOut.TreeWithEnd)
    f.write("\n")
    f.write(str(action))
    f.write("\n")
    f.write(str(JavaOut.Nl))
    f.write("\n")
    f.close()


def BeamSearch(sess, Code_gen_model, Nl, N, NL_number):
    Javaout = getJavaOut(Nl)
    global J_NlList
    J_NlList = Nl.strip().split()
    close_table = {}
    close_table[Javaout.Tree] = 1
    Beam = [Javaout]
    Set_ = Q.PriorityQueue()
    level = 0
    words = Nl.split()
    while True:
        level += 1
        Set_ = Q.PriorityQueue()
        if level > 10000:
            N -= 1
        for JavaOut in Beam:
            if JavaOut.is_end :
                Set_.put(JavaOut)
                continue
            print ("-----------")
            try:
                res = getAction(sess, Code_gen_model, JavaOut)
                list_res = [[res[i], i] for i in range(len(res))]
#               list_res = sorted(list_res, reverse=True)
                list_res = sorted(list_res, reverse=True)
            except:
                JavaOut.is_end = True
                Set_.put(JavaOut)
                continue
            count_n = N
            for t in range(len(list_res)):
                if t >= count_n:
                    break
                i = int(list_res[t][1])
                if i < len(Rule) and Rule[i][0] != JavaOut.Node:
                    count_n += 1
                    continue
                if i >= len(Rule) + len(words):
                    count_n += 1
                    continue
                if i >= len(Rule) and JavaOut.Node.strip() not in copy_node:
                    count_n += 1
                    continue
                
                WriteJavaIn(JavaOut, i )
                global global_step 
                global_step = JavaOut.gs
                global step_list_p
                step_list_p = deepcopy(JavaOut.step_list)
                J_run()
                JavaOutNext = getJavaOut(Nl)
                JavaOutNext.step_list = step_list_p
                JavaOutNext.gs = global_step
                if JavaOutNext.state == "error":
                    count_n += 1
                    continue
                JavaOutNext.RuleList = deepcopy(JavaOut.RuleList)
                JavaOutNext.Root = deepcopy(JavaOut.Root) + JavaOutNext.Root
                JavaOutNext.rule = deepcopy(JavaOut.rule)
                JavaOutNext.father_index = deepcopy(JavaOut.father_index)#.append(father_index_now)
                JavaOutNext.father_index.append(father_index_now)
                nowtree = JavaOutNext.Tree
                print (JavaOutNext.Tree)
                apa = 0.6
                if JavaOutNext.state == "grow":
                    print("grow")
                    print ("{Rule: %s}" % str(i))
                    if len(JavaOutNext.Tree.split()) > 1000:
                        continue
                    JavaOutNext.Probility = (JavaOut.Probility * math.pow(len(JavaOut.RuleList), apa) + math.log(max(1e-10, res[i]))) / math.pow(len(JavaOut.RuleList) + 1, apa)
                    JavaOutNext.RuleList.append(i + 1)
                    Set_.put(JavaOutNext)

                elif JavaOutNext.state == "end": # BUG!!!!?????
                    if JavaOutNext.Tree != JavaOut.Tree:
                        JavaOutNext.Probility = (JavaOut.Probility * math.pow(len(JavaOut.RuleList), apa) + math.log(max(1e-10, res[i]))) / math.pow(len(JavaOut.RuleList) + 1, apa)
                    else:
                        JavaOutNext.Probility = JavaOut.Probility
                    JavaOutNext.is_end = True
                    Set_.put(JavaOutNext)

        Beam = []
        endnum = 0

        while((not Set_.empty()) and N > len(Beam)):
            JavaOut = Set_.get()
            print(JavaOut.Probility)
            close_table[JavaOut.Tree] = 1
            Beam.append(JavaOut)
            if JavaOut.is_end:
                endnum += 1
        
        if endnum >= N:
            f = open(project + "out/"+str(NL_number)+".txt","w")
            for JavaOut in Beam:
                f.write(JavaOut.Tree)
                f.write("\n")
                f.write(str(JavaOut.Probility))
                f.write("\n")
            f.close()
            break


def predict():
    global Tree_vocabu_size
    global NL_vocabu_size
    NL_vocabu_size = len(vocabulary)
    Tree_vocabu_size = len(tree_vocabulary)

    Code_gen_model = code_gen_model(classnum, embedding_size, conv_layernum, conv_layersize, rnn_layernum,
                                    batch_size, NL_vocabu_size, Tree_vocabu_size, NL_len, Tree_len, parent_len, learning_rate, keep_prob, len(char_vocabulary), rules_len)
    config = tf.ConfigProto(device_count={"GPU": 0})
    #config = tf.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        create_model(sess, "", "")
        f = open(project + "input.txt", "r")
        lines = f.readlines()
        f.close()

        for i in range(len(lines)):
            Nl = lines[i].strip()        
            print(Nl)
            f = open(project + "Tree_Feature.out", "w")
            f.write("root ^")
            f.write("\n")
            f.write("root")
            f.write("\n")
            f.write("Unknown")
            f.write("\n")
            f.write("root\n")
            f.write("root node_gen ^ ^\n")
            f.write("root node_gen ^ ^\n")
            f.write("Unknown root ^ ^\n")
            f.write("Unknown Unknown ^ ^\n")
            f.write("root node_gen ^\n")
            f.write("Unknown root ^ ^\n")
            f.write("Unknown Unknown ^ ^\n")
            f.close()
            BeamSearch(sess, Code_gen_model, Nl, int(sys.argv[2]), i)
            print(str(i) + "th code is finished")
def read_copy_node():
    f = open(project + "copy_node.txt", "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        copy_node.append(line.strip())


def main():
    read_copy_node()
    print ("predict start")
    predict()

main()
