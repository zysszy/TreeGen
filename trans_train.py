f = open("train.txt", "r")
lines = f.readlines()
f.close()

f = open("train_trans.txt", "w")

rule = []

def find_father(words, i):
    index = i - 1
    count = 0
    while index >= 0:
        #print ("fa", words[i])
        if "^" in words[index]:
            count -= 1
        else:
            count += 1
        if count == 1:
#            exit()
            print (words[index])
            return rule[index]
        index -= 1
    return -1

def find_father_index(words, i):
    index = i - 1
    count = 0
    while index >= 0:
        #print ("fa", words[i])
        if "^" in words[index]:
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

def get_deepth(line, rule, index_rule):
    words = line.strip().split()
    #depth = len(words)
    #if words[0] == "arguments":
    #    return "a" + str(depth)
    #return depth
    deepth = -1
    bf = ""
    for i in range(len(words)):
        word = words[i]
        if "^" in word:
            deepth -= 1
        else:
            if "node_gen" == words[i]:
            #    if bf == "arguments":
            #        return "a" + str(deepth)
                rule.append(-1)
                print ("-----")
                print (words[i])
                index = find_father_index(words, i)
                rule[index] = index_rule
                return find_father(words, index)
                #return father
            deepth += 1
        bf = word
        if i >= len(rule):
            rule.append(-1)
            #if "node_gen" == word:
            #    return deepth
    return -1




def write_one(lines, index):
    line_now = ""
    ast = ""
    deepth = []
    numbers = 1
    for i in range(9):
        if i == 5:
            line_now = lines[index + i].strip() + " " + str(1 + int(lines[index + i + 1].strip())) + "\n"
            numbers = line_now.strip().split()
            for tt in range(len(numbers)):
                numbers[tt] = str(int(numbers[tt]))
            ll = " ".join(numbers)
            f.write(ll + "\n")
        elif i == 6:
            rule_r = line_now.strip().split()
            l = []
            for r in rule_r:
                l.append(int(r))
                f.write(str(int(r) - 1))
                f.write(" ")
            f.write("\n")
        elif i == 7:
            continue
        else:
            f.write(lines[index + i].strip() + "\n")
            if i == 7:
                length = len(numbers)
                global rule
                rule = []
                index_rule = 0
                #for site in range(length)[::-1]:
                # if i == 7:
                #    deepth.append(get_deepth(lines[index + i - 6 - 8 * site], rule, index_rule))
                #    index_rule += 1
                #print (deepth)
#                exit()
            if i == 8:
                length = len(numbers)
                for site in range(length)[::-1]:
                    if i == 8:
                        deepth.append(int(lines[index + i - 9 * site]))
                        #index_rule += 1 
                #deepth.append(int(lines[index + i]))
    #deepth = deepth[::-1]
    for i in deepth:
        f.write(str(i) + " ")
    f.write("\n")



for i in range(10, len(lines) + 9):
    t = i % 9
    if t == 5: # predicted rules lines
       if i >= len(lines):
           rules = []
       else:
           rules = lines[i].strip().split()
       if len(rules) == 0: # 
           write_one(lines, i - 9 - 5)
           #rules = lines[i - 8].strip().split()

f.close()

f = open("dev.txt", "r")
lines = f.readlines()
f.close()

f = open("dev_trans.txt", "w")


for i in range(10, len(lines) + 9):
    t = i % 9
    if t == 5: # predicted rules lines
       if i >= len(lines):
           rules = []
       else:
           rules = lines[i].strip().split()
       if len(rules) == 0: # 
           print (i)
           write_one(lines, i - 9 - 5)
           #rules = lines[i - 8].strip().split()

f.close()

f = open("test.txt", "r")
lines = f.readlines()
f.close()

f = open("test_trans.txt", "w")


for i in range(10, len(lines) + 9):
    t = i % 9
    if t == 5: # predicted rules lines
       if i >= len(lines):
           rules = []
       else:
           rules = lines[i].strip().split()
       if len(rules) == 0: # 
           print (i)
           write_one(lines, i - 9 - 5)
           #rules = lines[i - 8].strip().split()

f.close()
