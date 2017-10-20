import os
from subprocess import Popen

def main():
	#print "hi"
	file_list = []
	for path, subdirs, files in os.walk('.'):
		for name in files:
			if '.' in name:
				continue
			#print os.path.join(path, name)
			file_list.append(os.path.join(path, name))
	
	#print file_list
	for item in file_list:
		name = item.split('\\')[-1]
		print name
		tmp = item
		tmp = "C:\\Users\\Zhengzi\\Desktop\\QiaoYang_Work\\x86-binaries-master" + tmp[1:]
		#print tmp
		
		execute_command = "extract_func.py" + " " + name
		popen_command = "idaq -c -A -Sextract.py " + tmp
		#popen_command = "idaq -c -A -S\"" + execute_command +"\" " + tmp
		#print popen_command
		
		p = Popen(popen_command)
		stdout, stderr = p.communicate()
			
			
if __name__ == '__main__':
	main()