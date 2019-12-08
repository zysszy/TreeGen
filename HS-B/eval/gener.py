for i in range(0,66):
    f = open("../out/"+str(i)+".txt","r")
    pro = str(f.readline())
    f.close()
    t = i
    code = pro.split()
    ans = " "
    first = True
    for i in range(len(code)):
        if "Card" in code[i]:
            if not first and "Query" not in code[i]:
                code[i] = code[i].replace("Card", "")
            first = False
        ans += " " + code[i]
    i = t
    ans = ans.replace("  ","")
    f = open(str(i + 1) + ".txt", "w")
    f.write(ans)
    f.close()
