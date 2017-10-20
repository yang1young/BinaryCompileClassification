import os
import subprocess

origin_path = "/home/qiaoyang/codeData/molili/ProgramData/"
new_path = "/home/qiaoyang/bisheData/binary_source_nohead/"
binary_path = "/home/qiaoyang/bisheData/binary_out/"


head = "#include <iostream>\n#include <stdio.h>\n#include <memory.h>\n" \
       "#include <string.h>\n#include <math.h>\n#include <malloc.h>\n" \
       "#include <stdlib.h>\n#include <iomanip>\n#include <algorithm>\n" \
       "using namespace std;\n#define PI 3.14159265\n" \
       "\n\n"

def reformate(origin_path,new_path,need_head):
    dirs = os.listdir(origin_path)
    dirs.sort(key = lambda x:int(x))
    for dir in dirs:
        files = os.listdir(origin_path+dir)
        files.sort(key=lambda x: int(str(x).split('.')[0]))
        if (not os.path.isdir(new_path + dir)):
            os.mkdir(new_path + dir)
        if (not os.path.isdir(binary_path + dir)):
            os.mkdir(binary_path + dir)
        count = 0
        for file in files:
            with open(origin_path+dir+'/'+file,'r') as f_read:
                code = f_read.read()
                code = code.replace("void main()","int main()")
                code = code.replace("void main ()","int main()")
            with open(new_path+dir+'/'+str(count)+'.cpp','w') as f_write:
                if(need_head):
                    f_write.write(head+code)
                else:
                    f_write.write(code)
            count+=1

def compile(path):
    total = list()
    dirs = os.listdir(path)
    dirs.sort(key=lambda x: int(x))
    for dir in dirs:
        files = os.listdir(path + dir)
        files.sort(key=lambda x: int(str(x).split('.')[0]))
        for file in files:
            full_source = path+dir+'/'+file
            full_binary = binary_path+dir+'/'+file.split('.')[0]+'.out'
            retcode = subprocess.call(["g++",full_source, "-g","-o",full_binary])
            print dir+'----'+file+'----'+str(retcode)
            if(retcode==1 and os.path.exists(full_binary)):
                os.remove(full_binary)
                print file+"  removed"
        counts = len(os.listdir(binary_path+dir))
        total.append(counts)
    i = 0
    for l in total:
        print str(i)+'--'+str(l)
        i+=1


reformate(origin_path,new_path,False)
#compile(new_path)