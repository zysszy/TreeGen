def read_tree (lines, file_name):
    f = open(file_name, "w")
    for i in range(len(lines)):
        t = i % 9
        if t == 4:
            f.write(lines[i])
    f.close()



f = open("train.txt", "r")
lines = f.readlines()
f.close()
read_tree(lines, "train_tree.txt")

f = open("dev.txt", "r")
lines = f.readlines()
f.close()
read_tree(lines, "dev_tree.txt")

f = open("test.txt", "r")
lines = f.readlines()
f.close()
read_tree(lines, "test_tree.txt")

