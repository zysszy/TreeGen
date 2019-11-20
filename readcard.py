import sys

#card_number = 0

def loadcardnum():
    print ("load card...")

    file = open (sys.argv[2] + ".txt", "r")
    lines = file.readlines()
    file.close()
    file = open("nlnum.txt", "w")
    f = open("copylst.txt","w")
    count = 0
    l = []
    copylst = []
    for i in range(len(lines)):
        if i % 9 == 5:
            #if lines[i].strip() == "0":
            if len(lines[i].strip()) == 0:#i >= 8 and lines[i - 8].strip() != lines[i].strip():
                count += 1
            if len(lines[i].strip().split()) > 150:
                continue
            l.append(count)
        if i % 9 == 6:
            if(int(lines[i]) >= 10000):
                copylst.append(1)
            else:
                copylst.append(0)
    file.write(str(l))
    file.close()
    f.write(str(copylst))
    print (len(l))
    return count
    #global card_number
    #card_number = count

loadcardnum()
