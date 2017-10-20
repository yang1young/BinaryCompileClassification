from idautils import *
from idaapi import *

class BFunc:
	def __init__(self, name):
		self.in_program = None
		self.cfg = None
		self.bbs = []
		self.name = name
		self.start_address = None
		
	def add_bb(self, bbb):
		self.bbs.append(bbb)
		
	def print_func(self):
		for item in self.bbs:
			item.print_bb()

class BBasicBlock:
	def __init__(self):
		self.in_function = None
		self.start_address = None
		self.end_address = None
		self.binstrs = []
		
	def add_instr(self, binstr):
		self.binstrs.append(binstr)
	
		
	def print_bb(self):
		for item in self.binstrs:
			item.print_instr()

		
class BInstr:
	def __init__(self):
		self.in_basicblock = None
		self.address = None
		self.disasm = None
		self.bytes = None
	
		
	def print_instr(self):
		instr = '{:>10} | {:>18} | {:>0}'.format(self.address, self.bytes, self.disasm)
		print instr
##extract all functions
##0x80b1371L |                 55 | push    ebp
def dump_all_func():
	idaapi.autoWait()
	program = []
	for seg in Segments():
		if SegName(seg) == ".text":
			functions = Functions(seg)
			for func_ea in functions:
				name = GetFunctionName(func_ea)
				bfunc = BFunc(name)
				for bb in FlowChart(get_func(func_ea), flags=FC_PREDS):
					bbb = BBasicBlock()
					for head in Heads(bb.startEA,bb.endEA):
						if isCode(getFlags(head)):
							binstr = BInstr()
							
							next = NextHead(head, bb.endEA+1)
							length = 0
							if next:
								length = next - head
							else: 
								length = bb.endEA - head
							bytes = GetManyBytes(head, length, False)
							if bytes:
								bytes = bytes.encode('hex')
							disasm = GetDisasm(head)
							
							binstr.address = str(hex(head))
							binstr.disasm = disasm
							binstr.bytes = bytes
							bbb.add_instr(binstr)
					bfunc.add_bb(bbb)
				program.append(bfunc)
				
	return program
	
if __name__ == '__main__':
	idaapi.autoWait()
	program = dump_all_func()
	for i in range(10):
		program[i].print_func()
	#Exit(1)