import os
import random
import clean_utils as cu
path = "/home/qiaoyang/codeData/binary_code/newData2/"
#path = "/home/yang/data/"
TRAIN_PROJECT = path+"1/"
DEV_PROJECT = path+"3/"
TEST_PROJECT = path+"2/"
OUTPUT_PATH = path#+"train_repalce_number/"
MAX_LENGTH = 150

def get_data(project_path,output_path,output_name,sample_percent,only_command,need_replace_number,max_length):
    file_handler = open(output_path+output_name,'w')
    files = os.listdir(project_path)
    count_vect = [0,0,0,0]
    error_count = 0
    for file in files:
        keep = random.random()
        tag = (file.split('$')[0]).split("_")[3]
        if(tag.replace('O','').isdigit()):
           t = int(tag.replace('O', ''))
           if ((keep <= sample_percent[0] and t==0) or (keep <= sample_percent[1] and t==1)
               or (keep <= sample_percent[2] and t==2) or (keep <= sample_percent[3] and t==3)):
                count_vect[t] += 1
                codes = open(project_path + file, 'r').readlines()
                for code in codes:
                    assemble_code = code.split('@')[0]
                    assemble_code = cu.assemble_clean(assemble_code, need_replace_number, max_length,False)
                    if (only_command):
                        temps = assemble_code.split('$')
                        new_temps = []
                        for temp in temps:
                            temp = temp.strip()
                            new_temps.append(temp.split(' ')[0])
                        assemble_code = ' '.join(new_temps)

                    byte_code = code.split('@')[1]
                    byte_code = cu.bytecode_clean(byte_code, max_length,False)
                    if(assemble_code!='' and byte_code!='' and len(assemble_code)!=0 and len(byte_code)!=0):
                        file_handler.write(tag.replace('O', '') + '@' + assemble_code + '@' + byte_code + '\n')
        else:
            error_count+=1
            print error_count
    print count_vect
    file_handler.close()

def get_data_top_byte(project_path,output_path,output_name,sample_percent,only_command,need_replace_number,max_length):
    file_handler = open(output_path+output_name,'w')
    files = os.listdir(project_path)
    count_vect = [0,0,0,0]
    error_count = 0
    for file in files:
        keep = random.random()
        tag = (file.split('$')[0]).split("_")[3]
        if(tag.replace('O','').isdigit()):
           t = int(tag.replace('O', ''))
           if ((keep <= sample_percent[0] and t==0) or (keep <= sample_percent[1] and t==1)
               or (keep <= sample_percent[2] and t==2) or (keep <= sample_percent[3] and t==3)):
                count_vect[t] += 1
                codes = open(project_path + file, 'r').readlines()
                start_code = codes[0]
                end_code = codes[-1]

                def get_new_code(code,is_byte,need_reverse):
                    if(is_byte):
                        code = code.split('@')[1]
                        code = cu.bytecode_clean(code, max_length/2,need_reverse)
                    else:
                        code = code.split('@')[0]
                        code = cu.assemble_clean(code, need_replace_number, max_length/2,need_reverse)
                    return code
                assemble_code = get_new_code(start_code,False,False)+' ~ '+ get_new_code(end_code,False,True)
                byte_code = get_new_code(start_code,True,False)+' ~ '+ get_new_code(end_code,True,True)

                if(assemble_code!='' and byte_code!='' and len(assemble_code)!=0 and len(byte_code)!=0):
                    file_handler.write(tag.replace('O', '') + '@' + assemble_code + '@' + byte_code + '\n')
        else:
            error_count+=1
            print error_count
    print count_vect
    file_handler.close()

if __name__ == "__main__":
    need_replace_number = False
    # get_data(TRAIN_PROJECT,OUTPUT_PATH,'data.train',[0.38,0.56,1,1],False,need_replace_number,MAX_LENGTH)
    # get_data(DEV_PROJECT,OUTPUT_PATH,'data.dev',[1,1,1,1],False,need_replace_number,MAX_LENGTH)
    # get_data(TEST_PROJECT,OUTPUT_PATH,'data.test',[0.5,0.5,0.5,0.5],False,need_replace_number,MAX_LENGTH)
    OUTPUT_PATH = OUTPUT_PATH+'top_byte/'
    get_data_top_byte(TRAIN_PROJECT, OUTPUT_PATH, 'data.train', [0.4, 0.56, 1, 1], False, need_replace_number, MAX_LENGTH)
    get_data_top_byte(DEV_PROJECT, OUTPUT_PATH, 'data.dev', [1, 1, 1, 1], False, need_replace_number, MAX_LENGTH)
    get_data_top_byte(TEST_PROJECT, OUTPUT_PATH, 'data.test', [0.5, 0.5, 0.5, 0.5], False, need_replace_number, MAX_LENGTH)

