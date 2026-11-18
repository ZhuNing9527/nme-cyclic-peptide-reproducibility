"""OpenMM 8.4 scoped periodic REST2 parameter setter.

The frozen transformation remains unchanged. This setter refreshes the cached
NonbondedForce dispersion coefficient after updating the matching parameter
defaults. The Context then uses that coefficient divided by its current volume.
Every lambda assignment, including resetting to 1, must use this setter.
It is not an openmmtools state/cache adapter; that is a subsequent gated step.
"""
import math
import openmm as mm
from common import require

class PeriodicREST2Setter:
    def __init__(self,system,context,frozen_module):
        require(mm.__version__ in ('8.4','8.4.0'),'SETTER_VALIDATED_ONLY_FOR_OPENMM_8_4')
        self.system=system;self.context=context;self.frozen=frozen_module
        nbs=[f for f in system.getForces() if isinstance(f,mm.NonbondedForce)]
        require(len(nbs)==1,'EXPECTED_ONE_NONBONDED_FORCE')
        self.nonbonded=nbs[0]
        require(self.nonbonded.getNonbondedMethod()==mm.NonbondedForce.PME,'EXPECTED_PME')
        require(self.nonbonded.getUseDispersionCorrection(),'EXPECTED_DISPERSION_CORRECTION')
        require(not self.nonbonded.getUseSwitchingFunction(),'UNVALIDATED_SWITCHING_FUNCTION')
        self.indices={self.nonbonded.getGlobalParameterName(i):i for i in range(self.nonbonded.getNumGlobalParameters())}
        require(set(self.indices)=={'rest2_lambda','rest2_sqrt_lambda'},'UNEXPECTED_GLOBAL_PARAMETERS')
        require(self.nonbonded.getNumParticleParameterOffsets()>0,'EXPECTED_DYNAMIC_OFFSETS')

    def set_lambda(self,value):
        values=self.frozen.lambda_parameters(value)
        # Check before mutation. On any later exception the caller must stop the job.
        require(all(math.isfinite(v) for v in values.values()),'NONFINITE_LAMBDA')
        self.frozen.set_context_lambda(self.context,value)
        # Keep every force's shared defaults consistent so serialized Systems can
        # construct a new Context at the current lambda without default conflicts.
        for force in self.system.getForces():
            if not hasattr(force,'getNumGlobalParameters'):continue
            for i in range(force.getNumGlobalParameters()):
                name=force.getGlobalParameterName(i)
                if name in values:force.setGlobalParameterDefaultValue(i,values[name])
        self.nonbonded.updateParametersInContext(self.context)
        actual=dict(self.context.getParameters())
        require(all(actual[k]==v for k,v in values.items()),'CONTEXT_LAMBDA_NOT_APPLIED')
        return values
