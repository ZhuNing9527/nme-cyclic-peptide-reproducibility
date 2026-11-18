from __future__ import annotations

import math

import openmm
from openmm import (
    CMMotionRemover,
    CustomAngleForce,
    CustomBondForce,
    CustomCompoundBondForce,
    CustomTorsionForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
    XmlSerializer,
)

from openmmtools.states import (
    CompoundThermodynamicState,
    GlobalParameterState,
    ThermodynamicState,
)


REST2_LAMBDA_NAME="rest2_lambda"
REST2_SQRT_LAMBDA_NAME="rest2_sqrt_lambda"


class REST2ParameterState(
    GlobalParameterState
):

    rest2_lambda=(
        GlobalParameterState.GlobalParameter(
            REST2_LAMBDA_NAME,
            standard_value=1.0,
        )
    )

    rest2_sqrt_lambda=(
        GlobalParameterState.GlobalParameter(
            REST2_SQRT_LAMBDA_NAME,
            standard_value=1.0,
        )
    )

    def set_lambda(
        self,
        value: float,
    ):
        value=float(
            value
        )

        if not (
            0.0
            <
            value
            <=
            1.0
        ):
            raise ValueError(
                "REST2 lambda must satisfy 0 < lambda <= 1"
            )

        self.rest2_lambda=value
        self.rest2_sqrt_lambda=math.sqrt(
            value
        )


def lambda_parameters(
    value: float,
):
    value=float(
        value
    )

    if not (
        0.0
        <
        value
        <=
        1.0
    ):
        raise ValueError(
            "REST2 lambda must satisfy 0 < lambda <= 1"
        )

    return {
        REST2_LAMBDA_NAME:
            value,

        REST2_SQRT_LAMBDA_NAME:
            math.sqrt(
                value
            ),
    }


def set_context_lambda(
    context,
    value: float,
):
    values=lambda_parameters(
        value
    )

    for name,parameter in values.items():
        context.setParameter(
            name,
            parameter,
        )


def make_compound_state(
    system,
    temperature,
    lambda_value: float,
):
    parameter_state=REST2ParameterState()

    parameter_state.set_lambda(
        lambda_value
    )

    thermo=ThermodynamicState(
        system=system,
        temperature=temperature,
    )

    return CompoundThermodynamicState(
        thermodynamic_state=thermo,
        composable_states=[
            parameter_state,
        ],
    )


def _copy_force_metadata(
    source,
    target,
):
    target.setForceGroup(
        source.getForceGroup()
    )

    target.setName(
        source.getName()
    )

    if (
        hasattr(
            source,
            "usesPeriodicBoundaryConditions",
        )
        and
        hasattr(
            target,
            "setUsesPeriodicBoundaryConditions",
        )
    ):
        target.setUsesPeriodicBoundaryConditions(
            source.usesPeriodicBoundaryConditions()
        )


def _assert_no_parameter_offsets(
    force: NonbondedForce,
):
    if force.getNumParticleParameterOffsets()!=0:
        raise RuntimeError(
            "REST2 v1 requires zero pre-existing NonbondedForce particle offsets"
        )

    if force.getNumExceptionParameterOffsets()!=0:
        raise RuntimeError(
            "REST2 v1 requires zero pre-existing NonbondedForce exception offsets"
        )


def _add_nonbonded_globals(
    force: NonbondedForce,
):
    names=[
        force.getGlobalParameterName(i)
        for i in range(
            force.getNumGlobalParameters()
        )
    ]

    for name in (
        REST2_LAMBDA_NAME,
        REST2_SQRT_LAMBDA_NAME,
    ):
        if name in names:
            raise RuntimeError(
                "REST2 parameter already present before transformation: "
                +name
            )

    force.addGlobalParameter(
        REST2_LAMBDA_NAME,
        1.0,
    )

    force.addGlobalParameter(
        REST2_SQRT_LAMBDA_NAME,
        1.0,
    )


def apply_rest2_scaling(
    source_system,
    solute_atoms,
    *,
    m4_force_group: int=31,
):
    """
    Return a deep-copied OpenMM System implementing:

        U(lambda)
          = U_bond + U_angle
          + lambda * (U_torsion_ss + U_nonbonded_ss + U_M4)
          + sqrt(lambda) * U_nonbonded_sv
          + U_vv_remaining

    for an explicit-solvent Amber-style System.

    The source System is never mutated.

    Supported energetic force classes:
      HarmonicBondForce
      HarmonicAngleForce
      PeriodicTorsionForce
      NonbondedForce
      CustomCompoundBondForce in m4_force_group

    CMMotionRemover is carried through unchanged.

    Constraints remain geometrical constraints and are not scaled.
    """

    system=XmlSerializer.deserialize(
        XmlSerializer.serialize(
            source_system
        )
    )

    n_particles=system.getNumParticles()

    solute=set(
        int(i)
        for i in solute_atoms
    )

    if not solute:
        raise ValueError(
            "solute atom set is empty"
        )

    if min(solute)<0 or max(solute)>=n_particles:
        raise ValueError(
            "solute atom index outside System"
        )


    audit={
        "n_particles":
            n_particles,

        "n_solute_atoms":
            len(
                solute
            ),

        "scaled_bonds":
            0,

        "unscaled_bonds":
            0,

        "scaled_angles":
            0,

        "unscaled_angles":
            0,

        "scaled_torsions":
            0,

        "unscaled_torsions":
            0,

        "nonbonded_particles_scaled":
            0,

        "solute_solute_exceptions_scaled_lambda":
            0,

        "solute_solvent_exceptions_scaled_sqrt_lambda":
            0,

        "solvent_solvent_exceptions_unscaled":
            0,

        "m4_forces_scaled":
            0,

        "unsupported_force_types":[],
    }


    removals=[]
    replacements=[]


    for force_index in range(
        system.getNumForces()
    ):
        force=system.getForce(
            force_index
        )


        if isinstance(
            force,
            HarmonicBondForce,
        ):
            # Practical REST2 / partial_tempering:
            # bond-stretch terms are deliberately NOT scaled.
            for term in range(
                force.getNumBonds()
            ):
                i,j,r0,k=force.getBondParameters(
                    term
                )

                i_solute=int(i) in solute
                j_solute=int(j) in solute

                if i_solute != j_solute:
                    raise RuntimeError(
                        "mixed solute-solvent covalent bond encountered"
                    )

                audit[
                    "unscaled_bonds"
                ]+=1

            continue

        if isinstance(
            force,
            HarmonicAngleForce,
        ):
            # Practical REST2 / partial_tempering:
            # angle-bending terms are deliberately NOT scaled.
            for term in range(
                force.getNumAngles()
            ):
                i,j,kidx,theta0,k=force.getAngleParameters(
                    term
                )

                flags=[
                    int(i) in solute,
                    int(j) in solute,
                    int(kidx) in solute,
                ]

                if any(flags) and not all(flags):
                    raise RuntimeError(
                        "mixed solute-solvent covalent angle encountered"
                    )

                audit[
                    "unscaled_angles"
                ]+=1

            continue

        if isinstance(
            force,
            PeriodicTorsionForce,
        ):
            solvent_force=PeriodicTorsionForce()

            scaled_force=CustomTorsionForce(
                REST2_LAMBDA_NAME
                +"*k*(1+cos(periodicity*theta-phase))"
            )

            scaled_force.addGlobalParameter(
                REST2_LAMBDA_NAME,
                1.0,
            )

            scaled_force.addPerTorsionParameter(
                "periodicity"
            )

            scaled_force.addPerTorsionParameter(
                "phase"
            )

            scaled_force.addPerTorsionParameter(
                "k"
            )

            _copy_force_metadata(
                force,
                solvent_force,
            )

            _copy_force_metadata(
                force,
                scaled_force,
            )

            solvent_count=0
            scaled_count=0

            for term in range(
                force.getNumTorsions()
            ):
                i,j,kidx,l,periodicity,phase,k=(
                    force.getTorsionParameters(
                        term
                    )
                )

                flags=[
                    int(i) in solute,
                    int(j) in solute,
                    int(kidx) in solute,
                    int(l) in solute,
                ]

                if all(
                    flags
                ):
                    scaled_force.addTorsion(
                        int(i),
                        int(j),
                        int(kidx),
                        int(l),
                        [
                            float(
                                periodicity
                            ),
                            phase,
                            k,
                        ],
                    )

                    scaled_count+=1

                elif not any(
                    flags
                ):
                    solvent_force.addTorsion(
                        int(i),
                        int(j),
                        int(kidx),
                        int(l),
                        periodicity,
                        phase,
                        k,
                    )

                    solvent_count+=1

                else:
                    raise RuntimeError(
                        "mixed solute-solvent covalent torsion encountered"
                    )

            removals.append(
                force_index
            )

            if solvent_count:
                replacements.append(
                    solvent_force
                )

            if scaled_count:
                replacements.append(
                    scaled_force
                )

            audit[
                "scaled_torsions"
            ]+=scaled_count

            audit[
                "unscaled_torsions"
            ]+=solvent_count

            continue


        if isinstance(
            force,
            NonbondedForce,
        ):
            _assert_no_parameter_offsets(
                force
            )

            _add_nonbonded_globals(
                force
            )

            for particle in range(
                force.getNumParticles()
            ):
                charge,sigma,epsilon=(
                    force.getParticleParameters(
                        particle
                    )
                )

                if particle not in solute:
                    continue

                force.setParticleParameters(
                    particle,
                    charge*0.0,
                    sigma,
                    epsilon*0.0,
                )

                force.addParticleParameterOffset(
                    REST2_SQRT_LAMBDA_NAME,
                    particle,
                    charge,
                    sigma*0.0,
                    epsilon*0.0,
                )

                force.addParticleParameterOffset(
                    REST2_LAMBDA_NAME,
                    particle,
                    charge*0.0,
                    sigma*0.0,
                    epsilon,
                )

                audit[
                    "nonbonded_particles_scaled"
                ]+=1


            for exception in range(
                force.getNumExceptions()
            ):
                i,j,chargeprod,sigma,epsilon=(
                    force.getExceptionParameters(
                        exception
                    )
                )

                i_solute=(
                    int(i) in solute
                )

                j_solute=(
                    int(j) in solute
                )

                if (
                    i_solute
                    and
                    j_solute
                ):
                    force.setExceptionParameters(
                        exception,
                        i,
                        j,
                        chargeprod*0.0,
                        sigma,
                        epsilon*0.0,
                    )

                    force.addExceptionParameterOffset(
                        REST2_LAMBDA_NAME,
                        exception,
                        chargeprod,
                        sigma*0.0,
                        epsilon,
                    )

                    audit[
                        "solute_solute_exceptions_scaled_lambda"
                    ]+=1

                elif (
                    i_solute
                    !=
                    j_solute
                ):
                    force.setExceptionParameters(
                        exception,
                        i,
                        j,
                        chargeprod*0.0,
                        sigma,
                        epsilon*0.0,
                    )

                    force.addExceptionParameterOffset(
                        REST2_SQRT_LAMBDA_NAME,
                        exception,
                        chargeprod,
                        sigma*0.0,
                        epsilon,
                    )

                    audit[
                        "solute_solvent_exceptions_scaled_sqrt_lambda"
                    ]+=1

                else:
                    audit[
                        "solvent_solvent_exceptions_unscaled"
                    ]+=1

            continue


        if isinstance(
            force,
            CustomCompoundBondForce,
        ):
            if force.getForceGroup()!=m4_force_group:
                raise RuntimeError(
                    "unrecognized CustomCompoundBondForce outside validated M4 force group"
                )

            existing=[
                force.getGlobalParameterName(i)
                for i in range(
                    force.getNumGlobalParameters()
                )
            ]

            if REST2_LAMBDA_NAME in existing:
                raise RuntimeError(
                    "M4 force already contains REST2 lambda before transformation"
                )

            original=force.getEnergyFunction()

            # CustomCompoundBondForce expressions may contain a main
            # energy expression followed by semicolon-separated
            # intermediate definitions.  REST2 must scale ONLY the
            # main energy while preserving definition ordering.
            pieces=original.split(";")

            main_expression=pieces[0].strip()

            definitions=[
                piece.strip()
                for piece in pieces[1:]
                if piece.strip()
            ]

            if not main_expression:
                raise RuntimeError(
                    "M4 CustomCompoundBondForce has empty main expression"
                )

            force.addGlobalParameter(
                REST2_LAMBDA_NAME,
                1.0,
            )

            scaled_expression=(
                REST2_LAMBDA_NAME
                +"*("
                +main_expression
                +")"
            )

            if definitions:
                scaled_expression+=(
                    ";"
                    +";".join(
                        definitions
                    )
                )

            force.setEnergyFunction(
                scaled_expression
            )

            audit[
                "m4_forces_scaled"
            ]+=1

            continue


        if isinstance(
            force,
            CMMotionRemover,
        ):
            continue


        audit[
            "unsupported_force_types"
        ].append(
            type(
                force
            ).__name__
        )


    if audit[
        "unsupported_force_types"
    ]:
        raise RuntimeError(
            "unsupported force classes for REST2 v1: "
            +repr(
                audit[
                    "unsupported_force_types"
                ]
            )
        )


    for index in sorted(
        removals,
        reverse=True,
    ):
        system.removeForce(
            index
        )


    for force in replacements:
        system.addForce(
            force
        )


    return (
        system,
        audit,
    )
