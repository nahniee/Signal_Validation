"""Execute these plain-Python workpapers in-process, retaining Jupyter outputs.
No network/kernel service is required. Stop immediately on any cell error.
"""
import ast, contextlib, io, os
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4)
    namespace={'__name__':'__main__'}
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code': continue
        count+=1; cell.execution_count=count; cell.outputs=[]
        tree=ast.parse(cell.source)
        last=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
        stdout=io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree,str(path),'exec'),namespace)
            result=eval(compile(ast.Expression(last.value),str(path),'eval'),namespace) if last else None
        if stdout.getvalue():
            cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=stdout.getvalue()))
        if result is not None:
            data={'text/plain':repr(result)}
            if hasattr(result,'_repr_html_'):
                html=result._repr_html_()
                if html: data['text/html']=html
            cell.outputs.append(nbformat.v4.new_output('execute_result',execution_count=count,data=data))
    nbformat.validate(nb)
    nbformat.write(nb,path)
    print(path.name, count, 'cells executed')
