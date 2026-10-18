"""Causal allocator stress: identical Adam graph, only state ownership differs."""
import gc,json,weakref
import torch
torch.set_num_threads(1)
def captured(retain):
    x=torch.nn.Parameter(torch.ones(4096,device='cuda'))
    opt=torch.optim.Adam([x],lr=.001,capturable=True)
    def update():
        opt.zero_grad(set_to_none=False);x.square().sum().backward();opt.step()
    s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):update()
    torch.cuda.current_stream().wait_stream(s)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):update()
    refs=[weakref.ref(v) for state in opt.state.values() for v in state.values() if torch.is_tensor(v)]
    pointers=[v.data_ptr() for state in opt.state.values() for v in state.values() if torch.is_tensor(v)]
    if retain:graph._giada_optimizers=[opt]
    return graph,x,refs,pointers,s
rows=[]
for retain in (False,True):
    graph,x,refs,pointers,stream=captured(retain);gc.collect();torch.cuda.synchronize()
    alive=[r() is not None for r in refs];trash=[]
    with torch.cuda.stream(stream):
        for _ in range(128):trash.append(torch.full((4096,),float('nan'),device='cuda'))
    torch.cuda.current_stream().wait_stream(stream)
    reused=sum(t.data_ptr() in pointers for t in trash)
    graph.replay();torch.cuda.synchronize()
    rows.append(dict(retain=retain,optimizer_state_alive=alive,reused_state_allocations=reused,parameters_finite=bool(torch.isfinite(x).all())))
    del trash,graph,x;gc.collect()
assert not any(rows[0]['optimizer_state_alive'])
assert all(rows[1]['optimizer_state_alive']) and rows[1]['parameters_finite']
assert rows[0]['reused_state_allocations']>0 and not rows[0]['parameters_finite'],rows
print(json.dumps(dict(valid=True,rows=rows,causal_factor='optimizer state lifetime only')),flush=True)
