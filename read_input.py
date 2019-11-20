f = open("test_trans.txt", "r")
lines = f.readlines()
f.close()

f = open("input.txt", "w")
for i in range(len(lines)):
    t = i % 9
    if t == 0:
        f.write(lines[i])
f.close()
