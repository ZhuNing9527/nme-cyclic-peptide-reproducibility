"""Composable REST2 state that refreshes the validated periodic tail cache."""
import math
import openmm as mm
from openmm import unit
from openmmtools import states,cache
from common import require,load_rest2
from periodic_rest2_context_v1 import PeriodicREST2Setter

class PeriodicREST2State(states.GlobalParameterState):
    rest2_lambda=states.GlobalParameterState.GlobalParameter('rest2_lambda',standard_value=1.0)
    rest2_sqrt_lambda=states.GlobalParameterState.GlobalParameter('rest2_sqrt_lambda',standard_value=1.0)

    def set_lambda(self,value):
        value=float(value);require(math.isfinite(value) and 0<value<=1,'INVALID_REST2_LAMBDA')
        self.rest2_lambda=value;self.rest2_sqrt_lambda=math.sqrt(value)

    def _checked_lambda(self):
        value=self.rest2_lambda;root=self.rest2_sqrt_lambda
        require(value is not None and root is not None and math.isfinite(value) and 0<value<=1,'UNDEFINED_OR_INVALID_REST2_STATE')
        require(abs(root-math.sqrt(value))<1e-14,'INCONSISTENT_REST2_PARAMETER_PAIR')
        return value

    def apply_to_system(self,system):
        self._checked_lambda();super().apply_to_system(system)

    def apply_to_context(self,context):
        value=self._checked_lambda()
        # Called by both CompoundThermodynamicState.apply_to_context and its
        # optimized _apply_to_context_in_state path. No manual cache bypass.
        PeriodicREST2Setter(context.getSystem(),context,load_rest2()).set_lambda(value)

def make_states(physical,nsolute,values):
    transformed,audit=load_rest2().apply_rest2_scaling(physical,range(nsolute))
    thermodynamic=states.ThermodynamicState(transformed,temperature=297*unit.kelvin)
    out=[]
    for value in values:
        parameter=PeriodicREST2State();parameter.set_lambda(value)
        out.append(states.CompoundThermodynamicState(thermodynamic,[parameter]))
    require(all(out[0].is_state_compatible(x) for x in out),'LAMBDA_STATES_NOT_CACHE_COMPATIBLE')
    return out,audit

class OneContextCache(cache.ContextCache):
    """Capacity one plus a hard check that no second Context is ever constructed."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,capacity=1,**kwargs);self.context_ids=set();self.requests=0
    def get_context(self,*args,**kwargs):
        state=args[0] if args else kwargs['thermodynamic_state']
        requested_integrator=args[1] if len(args)>1 else kwargs.get('integrator')
        state_id=self._generate_state_id(state)
        integrator_id=self._generate_integrator_id(requested_integrator) if requested_integrator is not None else None
        if self.requests:
            require(state_id==self.initial_state_id,'INCOMPATIBLE_STATE_BLOCKED_BEFORE_CONTEXT_CREATION')
            if integrator_id is not None:require(integrator_id==self.initial_integrator_id,'INCOMPATIBLE_INTEGRATOR_BLOCKED_BEFORE_CONTEXT_CREATION')
        context,integrator=super().get_context(*args,**kwargs)
        if not self.requests:
            self.initial_state_id=state_id;self.initial_integrator_id=self._generate_integrator_id(integrator)
        self.context_ids.add(int(context.this));self.requests+=1
        require(len(self.context_ids)==1 and len(self)==1,'UNEXPECTED_SECOND_CONTEXT')
        return context,integrator
