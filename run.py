#-*-coding:utf-8-*-
import sys
project = str(sys.argv[1]) + "/"
from code_generate_model import *
from resolve_data import *
import os
import tensorflow as tf
import numpy as np
import os
import math
import queue as Q
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"]="0"

vocabu = {}
tree_vocabu = {}
vocabu_func = {}
tree_vocabu_func = {}
vocabu_var = {}
tree_vocabu_var = {}

embedding_size = 128
conv_layernum = 128
conv_layersize = 3
rnn_layernum = 50
batch_size = 80
NL_vocabu_size = len(vocabulary)
Tree_vocabu_size = len(tree_vocabulary)
NL_len = nl_len
Tree_len = tree_len
learning_rate = 1e-4
keep_prob = 0.8
pretrain_times = 100000
pretrain_dis_times = 2
train_times = 1000
parent_len = 20
rule_num_len = 1350

rules_len = rulelist_len = 150

numberstack = []
list2wordlist = []
cardnum = []
copynum = 0
copylst = []

def pre_mask():
    mask = np.zeros([rulelist_len, rulelist_len])
    for i in range(rulelist_len):
        for t in range(i + 1):
            mask[i][t] = 1
    return mask

def get_card(lst):
    global cardnum
    global copynum
    global copylst
    if True:#len(cardnum) == 0:
        f = open(project + "nlnum.txt", "r")
        st = f.read()
        cardnum = eval(st)
        f.close()
    if True:#copynum == 0:
        f = open(project + "copylst.txt", "r")
        st = f.read()
        copylst = eval(st)
        for x in copylst:
          if x == 1:
            copynum += 1
    
    dic = {}
    copydic = {}
    wrongnum = 0
    wrongcnum = 0
    for i, x in enumerate(lst):
        if x == False:
          if copylst[i] == 1:
            wrongnum += 1
            if cardnum[i] not in copydic:
              wrongcnum += 1
              copydic[cardnum[i]] = 1
          if cardnum[i] not in dic:
            dic[cardnum[i]] = 1
    return 491-len(dic), wrongnum/copynum, wrongcnum

def create_model(session, g, placeholder=""):
    if(os.path.exists(project + "save1")):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(project + "save1/"))
        print("load the model")
    else:
        session.run(tf.global_variables_initializer(), feed_dict={})
        print("create a new model")


def save_model(session, number):
    saver = tf.train.Saver()
    saver.save(session, project + "save" + str(number) + "/model.cpkt")

def save_model_time(session, number, card):
    saver = tf.train.Saver()
    saver.save(session, project + "save_list/save" + str(number) + "_" + str(card) + "/model.cpkt")

def get_state(batch_data):
    vec = np.zeros([len(batch_data[6])])
    for i in range(len(batch_data[6])):
        index = 79
        for t in range(len(batch_data[6][i])):
            if batch_data[6][i][t] == 0:
                index = t - 1
                break
        vec[i] = index
    return vec

def g_pretrain(sess, model, batch_data):
    batch = deepcopy(batch_data)
    rewards = np.zeros([len(batch[1])])

    for i in range(len(rewards)):
        rewards[i] = 1

    loss_mask = np.zeros([len(batch[9]), len(batch[9][0])])
    for i in range(len(batch[9])):
        loss_mask[i][0] = 1
        for t in range(1, len(batch[9][i])):
            if batch[9][i][t] == 0:
                break
            loss_mask[i][t] = 1

    state = get_state(batch_data)
    
    _, pre, a = sess.run([model.optim, model.correct_prediction, model.cross_entropy], feed_dict={model.input_NL: batch[0],
                                                  model.input_NLChar:batch[1],
                                                  model.inputparentlist: batch[5],
                                                  model.inputrulelist:batch[6],
                                                  model.inputrulelistnode:batch[7],
                                                  model.inputrulelistson:batch[8],
                                                  model.inputY_Num: batch[9],
                                                  model.tree_path_vec: batch[12],
                                                  model.labels:batch[18],
                                                  model.loss_mask:loss_mask,
                                                  model.antimask: pre_mask(),
                                                  model.treemask: batch[16],
                                                  model.father_mat:batch[17],
                                                  model.state:state,
                                                  model.keep_prob: 0.85,
                                                  model.rewards: rewards,
                                                  model.is_train: True
                                                  })
    return pre

def rules_component_batch(batch):
    vecnode = np.zeros([len(batch[2]), len(batch[2][0])])
    vecson = np.zeros([len(batch[2]), len(batch[2][0]), 3])
    #print (batch[2])
    #print (batch[-1])
    for i in range(len(batch[2])):
        for t in range(len(batch[2][0])):
            vnode, vson = rulebondast(int(batch[2][i][t]), "", batch[-1][i])
            vecnode[i, t] = vnode[0]
            for q in range(3):
                vecson[i, t, q] = vson[q]
    return [vecnode, vecson]

def g_eval(sess, model, batch_data):
    batch = batch_data
    rewards = np.zeros([len(batch[1])])

    for i in range(len(rewards)):
        rewards[i] = 1

    loss_mask = np.zeros([len(batch[9]), len(batch[9][0])])
    for i in range(len(batch[9])):
        loss_mask[i][0] = 1
        for t in range(1, len(batch[9][i])):
            if batch[9][i][t] == 0:
                break
            loss_mask[i][t] = 1
    
    state = get_state(batch_data)
    acc, pre, pre_rules = sess.run([model.accuracy, model.correct_prediction, model.max_res], feed_dict={model.input_NL: batch[0],
                                                model.input_NLChar:batch[1],
                                                model.inputparentlist: batch[5],
                                                model.inputrulelist:batch[6],
                                                model.inputrulelistnode:batch[7],
                                                model.inputrulelistson:batch[8],
                                                model.inputY_Num: batch[9],
                                                model.tree_path_vec: batch[12],
                                                model.loss_mask:loss_mask,
                                                model.antimask:pre_mask(),
                                                model.treemask:batch[16],
                                                model.father_mat:batch[17],
                                                model.labels:batch[18],
                                                model.state:state,
                                                model.keep_prob: 1,
                                                model.rewards: rewards,
                                                model.is_train: False
                                                })  
    p = []
    max_res = []
    for i in range(len(batch[9])):
        for t in range(rules_len):
            if batch[6][i][t] != 0:
                p.append(pre[i][t])
                max_res.append(pre_rules[i][t])
            else:
                p.pop()
                max_res.pop()
                break
    return acc, p, max_res


def run():
    Code_gen_model = code_gen_model(classnum, embedding_size, conv_layernum, conv_layersize, rnn_layernum,
                                    batch_size, NL_vocabu_size, Tree_vocabu_size, NL_len, Tree_len, parent_len, learning_rate, keep_prob, len(char_vocabulary))
    valid_batch, _ = batch_data(batch_size, "dev") # read data 
    best_accuracy = 0
    best_card = 0
    config = tf.ConfigProto(allow_soft_placement=True)#, log_device_placement=True)
    config.gpu_options.allow_growth = True
    f = open(project + "out.txt", "w")
    with tf.Session(config=config) as sess:
        create_model(sess, Code_gen_model, "")
        best_time = -1
        for i in tqdm(range(pretrain_times)):
            Code_gen_model.steps += 1.
            batch, _ = batch_data(batch_size, "train")
            for j in tqdm(range(len(batch))):
                if i % 3 == 0 and j % 2000 == 0: #eval
                    ac = 0
                    res = []
                    sumac = 0
                    length = 0
                    for k in range(len(valid_batch)):
                        ac1, loss1, _ = g_eval(sess, Code_gen_model, valid_batch[k])

                        res.extend(loss1)
                        ac += ac1;
                    ac /= len(valid_batch)
                    card, copyc, copycard = get_card(res)
                    strs = str(ac) + " " + str(card) + "\n"
                    f.write(strs)
                    f.flush()

                    print("current accuracy " +
                          str(ac) + " string accuarcy is " + str(card))
                    if best_card < card:
                        best_card = card
                        best_accuracy = ac
                        save_model(sess, 1)
                        best_time = i
                        print("find the better accuracy " +
                              str(best_accuracy) + "in epoches " + str(i))
                    elif card == best_card:
                        if (best_accuracy < ac):
                            best_card = card
                            best_accuracy = ac
                            save_model(sess, 1)
                            print("find the better accuracy " +
                              str(best_accuracy) + "in epoches " + str(i))
                    
                if i % 50 == 0 and j == 0:
                    ac = 0
                    res = []
                    sumac = 0
                    length = 0
                    for k in range(len(valid_batch)):
                        ac1, loss1, _ = g_eval(sess, Code_gen_model, valid_batch[k])

                        res.extend(loss1)
                        ac += ac1;
                    print (len(res))
                    ac /= len(valid_batch)
                    card, copyc, copycard = get_card(res)

                    print("current accuracy " +
                          str(ac) + " string accuracy is " + str(card))
                    save_model_time(sess, i, str(int(Code_gen_model.steps)))
                g_pretrain(sess, Code_gen_model, batch[j])
                tf.train.global_step(sess, Code_gen_model.global_step)

    f.close()
    #print("training finish")
    return




def main():
    run()

main()
