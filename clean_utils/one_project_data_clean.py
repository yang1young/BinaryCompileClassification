import os
import random
import clean_utils.clean_utils as cu

DATA_DIR = ""
DATA_RAW = DATA_DIR+'raw/'
DIRS = ['0/', '1/', '2/', '3/']
OUTPUT_DIR = ""


def data_format(data_raw,out_dirs,dirs_name):
    files = os.listdir(data_raw)
    for file in files:
        tag = file.split("_")[3]
        dir = ''
        if(tag=='O0'):
            dir = dirs_name[0]
        elif(tag=='O1'):
            dir = dirs_name[1]
        elif (tag == 'O2'):
            dir = dirs_name[2]
        elif (tag == 'O3'):
            dir = dirs_name[3]
        if(dir!=''):
            # flag = random.random()
            # if(flag<0.2):
            #     temp_path = test_dir
            # else:
            #     temp_path = train_dir
            temp_file = open(out_dirs+ dir + file, 'w')
            temp_file.write(open(data_raw+file,'r').read())
            temp_file.close()


def get_sample(output_file_dir,code_file_name,need_replace_number,only_command,sample_percent):
    code_train_handler_sample = open(output_file_dir + code_file_name + '.train', 'w')
    code_dev_handler_sample = open(output_file_dir + code_file_name + '.dev', 'w')
    code_test_handler_sample = open(output_file_dir + code_file_name + '.test', 'w')
    for dir in DIRS:
        dir = DATA_DIR+dir
        files = os.listdir(dir)
        if(len(files)==0):
            continue
        tag = dir.split('/')[-2]
        print tag
        for file in files:
            keep = random.random()
            if(keep<sample_percent):
                flag = random.random()
                if (flag < 0.2):
                    handler = code_test_handler_sample
                elif(flag>0.2 and flag<0.3):
                    handler = code_dev_handler_sample
                else:
                    handler = code_train_handler_sample
                codes = open(dir+file,'r').readlines()
                for code in codes:
                    code = cu.assemble_clean(code,need_replace_number)
                    if(only_command):
                        temps = code.split('$')
                        new_temps =[]
                        for temp in temps:
                            temp = temp.strip()
                            new_temps.append(temp.split(' ')[0])
                        code = ' '.join(new_temps)
                    handler.write(tag+'@'+code+'\n')
    code_train_handler_sample.close()
    code_test_handler_sample.close()
    code_dev_handler_sample.close()


if __name__ == "__main__":
    print ''
