from pycparser import c_ast,parse_file

class FuncDefVisitor(c_ast.NodeVisitor):
	def __init__(self):
		self.fun_list=[]
		self.coord_list=[]
		
	def visit_FuncDef(self, node):
		self.fun_list.append(node.decl.name)
		self.coord_list.append(node.decl.coord)
	#return fun_list,coord_list
        

def get_func_name(file_name):
	ast = parse_file(file_name,use_cpp=False)
	v = FuncDefVisitor()
	v.visit(ast)
	return v.fun_list
