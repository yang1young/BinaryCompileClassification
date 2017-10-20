from idautils import *
from idaapi import *
import hashlib
import sys
import csv

def check_name(func_name, real_name):
	if func_name == real_name:
		return True
	return False

def dump_func_backup():
	for seg in Segments():
		if SegName(seg) == ".text":
			functions = Functions(seg)
			for func_ea in functions:
				name = GetFunctionName(func_ea)
				p_name = GetInputFile()
				
				mnen_list = []
				disasm_list = []
				bytes_list = []
				
				to_write = ""
				for bb in FlowChart(get_func(func_ea), flags=FC_PREDS):
					disasm_list = []
					bytes_list = []
					mnen_list_bb = []
					
					pre = bb.startEA
					for head in Heads(bb.startEA,bb.endEA):
						if isCode(getFlags(head)):
							next = NextHead(head, bb.endEA+1)
							
							length = 0
							if next:
								length = next - head
								
							else: 
								length = bb.endEA - head
								
							bytes = GetManyBytes(head, length, False)
							if bytes:
								bytes = bytes.encode('hex')
								bytes_list.append(bytes)
							
							disasm = GetDisasm(head)
							disasm = str(disasm).replace('$','').replace('@','')
							disasm_list.append(disasm)
							
							mnem = GetMnem(head)
							if mnem:
								mnen_list.append(str(mnem))
								mnen_list_bb.append(str(mnem))
					disasm_str = ' $ '.join(disasm_list) 
					bytes_str = ' $ '.join(bytes_list)
					
					m_bb = hashlib.md5()
					mnem_str_bb = ' '.join(mnen_list_bb) 
					m_bb.update(mnem_str_bb)
					md5_bb = str(m_bb.hexdigest())
					
					to_write += disasm_str
					to_write += ' @ '
					to_write += bytes_str
					to_write += ' @ '
					to_write += md5_bb
					to_write += '\n'
					
					
				m = hashlib.md5()
				mnem_str = ' '.join(mnen_list) 
				m.update(mnem_str)
				md5 = str(m.hexdigest())
				
				path_new = "C:\\Users\\Zhengzi\\Desktop\\open3\\"+p_name+"$"+name+"$" +str(hex(func_ea)) +"_" + md5 + ".rec"
				f = open(path_new, 'w')
				f.write(to_write)
				f.close()

def dump_func_map():
	function_map = []
	for seg in Segments():
		if SegName(seg) == ".text":
			functions = Functions(seg)
			
			for func_ea in functions:
				name = GetFunctionName(func_ea)
				#p_name = GetInputFile()
				mnen_list = []

				to_write = ""
				for bb in FlowChart(get_func(func_ea), flags=FC_PREDS):
					mnen_list_bb = []
					
					for head in Heads(bb.startEA,bb.endEA):
						if isCode(getFlags(head)):
							mnem = GetMnem(head)
							if mnem:
								mnen_list.append(str(mnem))
								mnen_list_bb.append(str(mnem))

				m = hashlib.md5()
				mnem_str = ' '.join(mnen_list) 
				m.update(mnem_str)
				md5 = str(m.hexdigest())
				
				function_map_record = [name, str(hex(func_ea)), md5]
				function_map.append(function_map_record)

	f = open('mapping.txt','wb')       
	wr = csv.writer(f, dialect='excel')
	wr.writerows(function_map)
	f.close()




if __name__ == '__main__':
	idaapi.autoWait()
	#dump_func()
	dump_func_map()
	Exit(1)
	#dump_func("CVE-2324-1231", "list_cipher_fn", "C:\\Users\\Zhengzi\\Desktop\\working\\test1\\")
