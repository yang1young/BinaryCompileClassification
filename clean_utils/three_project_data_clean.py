import os
import random
import clean_utils as cu
path = "/home/qiaoyang/codeData/binary_code/newData/"
TRAIN_PROJECT = path+"2/"
DEV_PROJECT = path+"3/"
TEST_PROJECT = path+"1/"
OUTPUT_PATH = path

def get_data(project_path,output_path,output_name,sample_percent,only_command,need_replace_number):
    file_handler = open(output_path+output_name,'w')
    files = os.listdir(project_path)
    for file in files:
        keep = random.random()
        if (keep < sample_percent):
            tag = file.split("_")[3]
            codes = open(project_path+file, 'r').readlines()
            for code in codes:
                assemble_code = code.split('@')[0]
                assemble_code = cu.assemble_clean(assemble_code, need_replace_number)
                if (only_command):
                    temps = assemble_code.split('$')
                    new_temps = []
                    for temp in temps:
                        temp = temp.strip()
                        new_temps.append(temp.split(' ')[0])
                    assemble_code = ' '.join(new_temps)

                byte_code = code.split('@')[1]
                byte_code = cu.bytecode_clean(byte_code)

                file_handler.write(tag.replace('O','') + '@' + assemble_code + '@' + byte_code + '\n')
    file_handler.close()

if __name__ == "__main__":
    get_data(TRAIN_PROJECT,OUTPUT_PATH,'data.train',0.5,False,False)
    get_data(DEV_PROJECT,OUTPUT_PATH,'data.dev',1,False,False)
    get_data(TEST_PROJECT,OUTPUT_PATH,'data.test',0.5,False,False)


