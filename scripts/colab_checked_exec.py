"""Official CLI adapter: fail closed when kernel errors do not affect CLI exit.

No authentication handling: delegate that to the installed official CLI.
Use an explicit named session and serialize commands within each session.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

def kernel_exit(records):
    errors=[o for r in records for o in r.get('outputs',[]) if o.get('output_type')=='error']
    for error in errors:
        if error.get('ename')=='SystemExit':
            try:code=int(error.get('evalue','1'))
            except ValueError:code=1
            if code==0:continue
            return code if 1<=code<=255 else 1
        return 1
    return 0

def main():
    args=sys.argv[1:]
    cli=str(Path(sys.executable).with_name('colab'))
    base=[cli,'--auth=oauth2']
    if not args or args[0]!='exec':return subprocess.run(base+args).returncode
    if '--help' in args or '-h' in args:return subprocess.run(base+args).returncode
    session=None
    for i,arg in enumerate(args):
        if arg in ('-s','--session') and i+1<len(args):session=args[i+1]
        elif arg.startswith('--session='):session=arg.split('=',1)[1]
    if not session:
        print('GIADA: exec requires an explicit -s SESSION',file=sys.stderr);return 2
    with tempfile.TemporaryDirectory(prefix='giada-colab-log-check-') as tmp:
        history=Path(tmp)/'history.jsonl'
        def records():
            r=subprocess.run(base+['log','-s',session,'-o',str(history)],capture_output=True,text=True)
            if r.returncode or not history.exists():raise RuntimeError('Cannot verify structured CLI history')
            return [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        try:before=records()
        except (RuntimeError,ValueError) as exc:print(str(exc),file=sys.stderr);return 3
        code=subprocess.run(base+args).returncode
        if code:return code
        try:after=records()
        except (RuntimeError,ValueError) as exc:print(str(exc),file=sys.stderr);return 3
        if after[:len(before)]!=before:
            print('GIADA: session history changed concurrently; result unverified',file=sys.stderr);return 3
        new=[r for r in after[len(before):] if r.get('event_type')=='execution']
        if not new:
            print('GIADA: no new execution receipt; result unverified',file=sys.stderr);return 3
        code=kernel_exit(new)
        if code:print(f'GIADA: remote kernel failure detected; exit={code}',file=sys.stderr)
        return code
if __name__=='__main__':raise SystemExit(main())
