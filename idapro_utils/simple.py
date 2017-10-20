from idapro.idautils import *
from idapro.idaapi import *
from idapro.idc import *

idaapi.autoWait()
for seg in Segments():
    if SegName(seg) == ".text":
        for func_ea in Functions(seg):

            #get function name
            name = GetFunctionName(func_ea)
            
            #set a file name
            file_name = name + ".tttt"
            
            #set the file_content to empty
            file_content = ""
            
            for bb in FlowChart(get_func(func_ea), flags=FC_PREDS):
            
                disasm_list = []
                bytes_list = []
                mnen_list = []
                
                for head in Heads(bb.startEA,bb.endEA):
                    if isCode(getFlags(head)):
                        
                        #get mnemonic
                        mnemonic = GetMnem(head)
                        
                        #get disassembly
                        disasm = GetDisasm(head)
                        
                        
                        #get bytes
                        next = NextHead(head, bb.endEA+1)
                        length = 0
                        if next:
                            length = next - head
                        else:
                            length = bb.endEA - head
                        bytes = GetManyBytes(head, length, False)
                        #get bytes
                        
                        
                        #store the info to the list
                        disasm_list.append(disasm)
                        mnen_list.append(mnemonic)
                        if bytes:
                            bytes = bytes.encode('hex')
                            bytes_list.append(bytes)
                        #store the info to the list
                
                #flatten the list to string
                disasm_str = ' $ '.join(disasm_list) 
                bytes_str = ' $ '.join(bytes_list)
                
                #update the file content
                file_content += disasm_str
                file_content += ' @ '
                file_content += bytes_str
                file_content += '\n'
            
            #create a file and open it
            f = open(file_name, 'w')
            f.write(file_content)
            f.close()
            
#exit ida pro
#Exit(1)