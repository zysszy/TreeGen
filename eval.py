import sys

project = str(sys.argv[1]) + "/"

class ast_node:
    def __init__(self, node):
        self.node = node
        self.son = []

    def __str__(self):
        if self.node == "End":
            return ""
        ret = ""
        for son in self.son:
            ret += " " + str(son)
        return (str(self.node) + ret + " ^").replace("  ", " ")

    def __lt__(self, other):
        return str(self) > str(other)

    def do_sort(self):
        if self.node.lower() in ["and", "or"]:
            self.son[0].son = sorted(self.son[0].son)


def parse_tree(line):
    words = line.strip().split()
    stack = []
    ans = ""
    for i in range(len(words)):
        node = words[i]
        if node == "^":
            #stack.pop()
            stack[-1].do_sort()
            if len(stack) == 1:
                ans = stack[0]
            stack.pop()
        else:
            new_node = ast_node(str(node))
            if len(stack) != 0:
                now_node = stack[-1]
                now_node.son.append(new_node)
                #now_node.do_sort()
            stack.append(new_node)
    return str(ans)

if "HS" in project:
    import os 
    os.system("cd HS-B/eval && python3 gener.py")
    os.system("cd HS-B/eval && python3 ast2code.py")
    exit()

f = open(project + "test_output_ast.txt", "r")
lines = f.readlines()
f.close()

count = 0
cards = len(lines)
for i in range(len(lines)):
    f = open(project + "out/" + str(i) + ".txt", "r")
    gens = f.readlines()
    if parse_tree("root " + lines[i] + " ^") == parse_tree(gens[0]):
        count += 1
    #elif "count" in parse_tree(gens[0]) and "count" in parse_tree(lines[i]):
    #    print (parse_tree(lines[i]))
    #    print (parse_tree(gens[0]))
    #    print ("*************")
    else:
        print ("Gold:", parse_tree("root " +lines[i] + " ^"))
        print ("Gen:", parse_tree(gens[0]))
        print ("-------")
    print ("%s / %s : %s, All: %s" % (str(count), str(i + 1), str(count / (i + 1)), str(count / cards)) )
#for i in range(5):
#    print (lines[-i])
#    print (str(parse_tree(lines[-i])))
#    print ("------")



