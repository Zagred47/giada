"""Resume pure evaluator stages without touching any scientific predicate.

AST changes are limited to IO cache initialization and loop guards; numerical
bodies and timing regions are retained verbatim from the original functions.
"""
import ast,inspect,json,textwrap,types

def compile_function(original,transform,extra=None):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    transform(tree.body[0]);ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__,**(extra or {}))
    exec(compile(tree,original.__code__.co_filename+'[stage-cache]','exec'),namespace)
    return namespace[original.__name__]

def guard_cost(original):
    def transform(fn):
        found=0
        for node in ast.walk(fn):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='results' for t in node.targets):
                assert isinstance(node.value,ast.List) and not node.value.elts
                node.value=ast.parse("json.loads((output/'cost.json').read_text()) if (output/'cost.json').exists() else []",mode='eval').body
                found+=1
            if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='scope':
                node.body.insert(0,ast.parse("if any(r['batch_per_seed']==batch and r['scope']==scope for r in results): continue").body[0]);found+=1
        assert found==2,('Unexpected benchmark source',found)
    return compile_function(original,transform)

def guard_core(original):
    def transform(fn):
        found=0
        for node in ast.walk(fn):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='devrows' for t in node.targets):
                node.value=ast.parse("json.loads((output/'development_gpu.json').read_text()) if (output/'development_gpu.json').exists() else {}",mode='eval').body;found+=1
            if isinstance(node,ast.For) and ast.unparse(node.target)=='(name, model)' and any(isinstance(c,ast.Attribute) and c.attr=='Scanner' for c in ast.walk(node)):
                node.body.insert(0,ast.parse('if name in devrows: continue').body[0]);found+=1
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='mkdir' and not node.args and not node.keywords:
                node.keywords=[ast.keyword(arg='exist_ok',value=ast.Constant(True))]
        assert found==2,('Unexpected development evaluator source',found)
    return compile_function(original,transform)

def guard_compact_cost(original):
    def transform(fn):
        found=0
        for node in ast.walk(fn):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in node.targets):
                assert isinstance(node.value,ast.List)
                node.value=ast.parse("json.loads((out/'compact_execution_cost.json').read_text()) if (out/'compact_execution_cost.json').exists() else []",mode='eval').body;found+=1
            if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='stable':
                node.body.insert(0,ast.parse("if any(r['batch']==batch and r['recentered_membrane']==stable for r in rows): continue").body[0]);found+=1
            if isinstance(node,ast.Call) and ast.unparse(node.func)=='types.FunctionType':
                assert len(node.args)==2 and not node.keywords
                node.keywords=[ast.keyword(arg='argdefs',value=ast.parse('t.make_core.__defaults__',mode='eval').body)];found+=1
        assert found==3,('Unexpected compact evaluator source',found)
    return compile_function(original,transform)

def install(h,ev,output):
    from resume_task41h_frozen_evaluation import install_completed_panel_cache
    reused=install_completed_panel_cache(output)
    original_causal=h.c.b.causal_evaluate
    def causal(torch,root,folder,models,*args):
        path=folder/'causal_results.json'
        if path.exists():
            data=json.loads(path.read_text())
            assert set(data)=={n+'/'+m for n in models for m in ('imposed','autonomous')}
            reused.append(dict(panel='historical',rows=len(data)))
            print('[GIADA Task41h] reusing complete historical regressions',flush=True)
            return data
        return original_causal(torch,root,folder,models,*args)
    h.c.b.causal_evaluate=causal
    original_recurrence=h.c.j.i.old.evaluate
    def recurrence(root,folder,config,*args):
        path=folder/'recurrence_evaluation.json'
        if path.exists():
            data=json.loads(path.read_text())
            for name in config['architectures']:
                for role in ('development','stress_diagnostic'):
                    assert sum(r['model']=='joint_w'+name and r['role']==role for r in data)==72
            reused.append(dict(panel=str(folder.name),rows=len(data)))
            return None
        return original_recurrence(root,folder,config,*args)
    h.c.j.i.old.evaluate=recurrence
    h.g.fe.evaluate=guard_core(h.g.fe.evaluate)
    ev.ee.e.benchmark=guard_cost(ev.ee.e.benchmark)
    original_execution=h.g.fe.execution_matrix
    def execution(torch,root,folder,models,*args):
        name=next(iter(models));path=folder/('execution_matrix_'+name+'.json')
        if path.exists():
            rows=json.loads(path.read_text())
            if len(rows)==6 and len({(r['batch_per_seed'],r['physics']) for r in rows})==6:return rows
        return original_execution(torch,root,folder,models,*args)
    h.g.fe.execution_matrix=execution
    original_numerical=ev.numerical_panels
    def numerical(torch,root,folder,models,*args):
        path=folder/'voltage_precision.json'
        if path.exists():
            rows=json.loads(path.read_text())
            if len(rows)==10*len(models):return rows
        return original_numerical(torch,root,folder,models,*args)
    ev.numerical_panels=numerical
    original_oracle=ev.oracle_stress
    def oracle(torch,root,folder,models,*args):
        path=folder/'coefficient_oracle_stress.json'
        if path.exists():
            report=json.loads(path.read_text())
            if len(report['rows'])==4*36*len(models):return report
        return original_oracle(torch,root,folder,models,*args)
    ev.oracle_stress=oracle
    ev.execution_cost=guard_compact_cost(ev.execution_cost)
    return reused
