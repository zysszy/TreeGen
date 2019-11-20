f = open("train.txt", "r")
lines = f.readlines()
dic1 = {}
dic2 = {}
dic3 = {} # char dict
for i in range(len(lines)):
    t = i % 8
    if t == 0 and (i + 8 >= len(lines) or lines[i + 8] != lines[i]):
        words = lines[i].split()
        for word in words:
            if "^" in word:
                continue
            if word in dic1:
                dic1[word] += 1
            else:
                dic1[word] = 1
            for c in word:
                if c in dic3:
                    dic3[c] += 1
                else:
                    dic3[c] = 1
    if t == 4 or ( t == 1 and (i + 7 >= len(lines) or lines[i + 7] != lines[i - 1])):
        words = lines[i].split()
        for word in words:
            if "^" in word:
                continue
            if word in dic2:
                dic2[word] += 1
            else:
                dic2[word] = 1
f.close()
def read2file(dic, filename):
    f = open(filename, "w")
    l = []
    for i in dic:
        l.append([dic[i], i])
    l = sorted(l, reverse=True)
    for i in range(len(l)):
        f.write(l[i][1] + " " + str(l[i][0]) + '\n')

read2file(dic1, "nl_voc.txt")
read2file(dic2, "tree_voc.txt")
read2file(dic3, "char_voc.txt")
