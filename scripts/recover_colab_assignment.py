"""Bind an already allocated runtime after assign timed out; official SDK only."""
import argparse,json
from colab_profile_cli import configure_profile
configure_profile()
from colab_cli.common import state
from colab_cli.state import SessionState
parser=argparse.ArgumentParser();parser.add_argument('--session',required=True);parser.add_argument('--endpoint',required=True);args=parser.parse_args()
matches=[a for a in state.client.list_assignments() if a.endpoint==args.endpoint]
assert len(matches)==1,'Exact existing runtime must be present'
a=matches[0];proxy=a.runtime_proxy_info
state.store.add(SessionState(name=args.session,token=proxy.token,url=proxy.url,endpoint=a.endpoint,
    token_expires_at=proxy.expires_at(),variant=a.variant.name,accelerator=a.accelerator.value,machine_shape=a.machine_shape.name))
state.history.log_event(args.session,'session_recovered_after_assign_timeout',{'endpoint':a.endpoint})
print(json.dumps(dict(valid=True,name=args.session,endpoint=a.endpoint,accelerator=a.accelerator.value,new_runtime_created=False)))
