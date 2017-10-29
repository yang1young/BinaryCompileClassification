import os
from subprocess import Popen,PIPE
from  clean_utils.query_idapro import query_ida

source_path = "/home/qiaoyang/bisheData/binary_source/"
binary_path = "/home/qiaoyang/bisheData/binary_out/"
source_nohead_path = '/home/qiaoyang/bisheData/binary_source_nohead/'
binary_result_path = '/home/qiaoyang/bisheData/binary_result/'
ida_path = ""
python_script_path = ""


def main():
    dirs = os.listdir(binary_path)
    dirs.sort(key=lambda x: int(x))
    for dir in dirs:
        files = os.listdir(binary_path + dir)
        files.sort(key=lambda x: int(str(x).split('.')[0]))
        if (not os.path.isdir(binary_result_path + dir)):
            os.mkdir(binary_result_path + dir)
        for f in files:
            file_name = f.split(".")[0]
            source_nohead_file = source_nohead_path+dir+'/'+file_name+'.cpp'

            cmd = ida_path + ' -c -A -S"%s %s %s" %s' % (python_script_path, binary_result_path+dir+'/', source_nohead_file, binary_path+dir+'/'+f)
            # print cmd
            p = Popen(cmd, shell=True, stdout=PIPE)