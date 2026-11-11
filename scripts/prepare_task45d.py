"""Finalize45d factorial contrasts; no new teacher output or source upload."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    p=copy.deepcopy(json.loads((ROOT/'experiments/task45c_preregistration_v1.json').read_text()))
    p.update(task='45d',schema_version='45d-optimization-support-execution-v1',coupled_arms=['none','C0','C1','A00','A01','A10','A11','Ncap','Nline','T0','T1','J','R41','R111'])
    p['fitting'].update(fit_seeds=[9454103,9454117,9454129],dev_seeds=[9454207,9454219,9454231],coefficient_seed=9454001,points=4096,checkpoints=[1,4,16,64],max_whitened_step_norm=2.)
    p['matrix']=dict(atomic_checkpoints=[250,1000,4000],batch=512,theta_lr=.01,trunk_lr=.00001,gradient_norm_per_seed_group=1.,held_ticks=[1,10,100,10000],support_seed=9454031,
        arms={k:dict(support=int(k[1]),adapt_features=bool(int(k[2]))) for k in ('A00','A01','A10','A11')},
        objective='Canonical offline coefficient logits plus held-input gate/current predictions at .1/1/10/1000ms; frozen-logit preservation for all untargeted states. No claim that held-input loss is full variable-voltage BPTT. Actual1000ms variable-path DEV at each saved budget.',
        parameterization='Existing parent voltage a/b optionally trainable, same64width; only c/d columns3,9,10,20,26,27 trainable through whitened residual coordinates. No added inference features or state. FP64 training residual arithmetic; FP32 actual materialization at DEV.',
        support='D0 previous45c split uniform nominal/tail grid. D1 half same grid, half causal replay inputs from fresh controlled long-relaxation, weak-return and hyperpolarization histories; same4096points and paired batches. DEV union covers both and is common to all arms.',
        nap='Same fixed features/coefficient objective, independent seed ridge solves. Ncap max step norm2 versus Nline FIT-only quadratic line search1/.5/.25; checkpoints1/4/16/64 along each trajectory. Existing45c N00 objective; no changed acceptance tolerance.',
        timing='T0/T1 same matched FIT histories, same gate/current-preservation rows; only T1 adds signed current residuals at canonical10% crossing neighbors. Original event evaluator remains authoritative. Stop refinements only at norm<1e-8 all seeds; later snapshots explicitly marked early-stopped.',
        nomination='Whole-model no-new-failure on common freshDEV vs frozenC1 across all prefixes. Rank target pass sum then worst target severity, then earlier step/order. No component mixing across changed trunks. J copies nominated actual model; fallbackC1. Coupled/stress never nominate.')
    p['fitting'].update(objective='45d main atomic objective, Nap quadratic and timing contrasts are defined under matrix. Legacy45c arm names are not active.',precision='Whitened FP64 training residuals and independent seed linear solves; actual FP32 same-class materialization at every DEV checkpoint. Inference precision unchanged.',sensitivities='Exact original45c causal recurrence sensitivities used only for paired T0/T1; Nap cap/line search solves the same fixed-feature quadratic; Axy uses paired gradient updates.',nomination=p['matrix']['nomination'])
    p['regression_guards']=[r.replace('All13 arms','All14 arms') for r in p['regression_guards']]
    p['hypotheses']=[dict(id=h,contrast=s) for h,s in [
        ('H1','Ncap vs Nline and1/4/16/64 budgets distinguish truncated optimization from persistent fixed-feature residual'),
        ('H2','A10-A00 and A11-A01 distinguish matched support at same feature policy'),
        ('H3','A01-A00 and A11-A10, projection residual and materialization discrepancy probe usefulness of existing feature adaptation'),
        ('H4','T1-T0 signed crossing-neighbor objective with identical functional rows; actual original timing and regressions decide'),
        ('H5','Frozen J dual vs donor_pruned vs shared_features; no teacher inference, exact parity required for equivalent classification'),
        ('H6','Separate CUDA/wall microprofiles for parent/donor/full/feature components; matched formula32 and formula64 resident benchmarks')]]
    p['runtime_budget']='Single private T4 worker10800s/outer12000s. One notebook, no automatic resubmission. Accuracy arms resident together; independent seed moments/clipping; grouped backward/optimizer per update. Fit checkpoints persisted before coupled phase. Native45b reused, no NEURON load.'
    p['cost']['repetitions']=30
    p['execution_scope']='Precision,dt,18states,information,analytic readout and membrane/synapses unchanged. Exact variants from existing41n implementation, requalified on45d outputs. Non-bitwise variants diagnostic only and require full original quality evaluation before any adoption.'
    p['limits']+=['45d is exploratory, no release/46/47 authorization. Training budget is not proof of convergence; explicit loss and residual probes mandatory.','No claim that shared-feature/trunk adaptation is free: offline training cost, all-channel changes and materialized errors reported.']
    write_json(ROOT/'experiments/task45d_preregistration_v1.json',p)
if __name__=='__main__':main()
