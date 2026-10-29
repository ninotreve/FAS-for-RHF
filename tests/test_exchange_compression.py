"""Signed history factorization and mandatory original-Fock residual validation."""
import numpy as np
import pytest
from hf_fas.fem import MolecularFEMModel, regression_molecules, SCFConfig, solve_molecular_scf
from hf_fas.fem.scf import compress_exchange_density, prepare_fock_action
from hf_fas.tensor import TensorMesh, TensorQ1Space, TensorCoulomb


def model(name='LiH'):
    space=TensorQ1Space(TensorMesh(4,4.))
    return MolecularFEMModel(space,regression_molecules()[name],TensorCoulomb(space))


def test_signed_density_factorization_retains_negative_DIIS_directions():
    m=model();a=m.initial_state();b=m.rebuild((a.orbitals[0]+.1*np.random.default_rng(3).normal(size=a.orbitals[0].shape),))
    p=np.concatenate([a.orbitals[0],b.orbitals[0]],axis=1)
    weights=np.repeat([1.3,-.3],a.orbitals[0].shape[1])
    q,f,report=compress_exchange_density(m.space,p,weights,1e-13)
    np.testing.assert_allclose((q*f)@q.T,(p*weights)@p.T,rtol=1e-11,atol=1e-13)
    assert report['negative_retained_occupations']>0


def test_duplicate_history_compacts_and_matches_original_Fock_action():
    m=model();state=m.initial_state();states=(state,)*6;weights=np.ones(6)/6
    plain=prepare_fock_action(m,states,weights,0)
    compact=prepare_fock_action(m,states,weights,0,compression_tolerance=1e-12)
    v=np.random.default_rng(9).normal(size=(m.space.n_orbital,3))
    np.testing.assert_allclose(compact(v),plain(v),rtol=2e-12,atol=2e-12)
    assert compact.compression['retained_columns']==state.orbitals[0].shape[1]
    assert compact.compression['input_columns']==6*state.orbitals[0].shape[1]
    np.testing.assert_array_equal(compact.full_action(v),plain(v))


@pytest.mark.parametrize('cutoff',[1e-12,10.])
def test_complete_SCF_checks_original_action_and_falls_back_if_compression_is_too_large(cutoff):
    m=model('He');cfg=SCFConfig(eigensolver='dense',tolerance=1e-10,exchange_compression_tolerance=cutoff)
    result=solve_molecular_scf(m,config=cfg)
    assert result.converged and result.final_residual<1e-10
    checked=[row['inner'][0]['exchange_compression'] for row in result.history if 'inner' in row]
    assert checked and m.work.values['mixed_fock_validation_applications']>=len(checked)
    if cutoff==10.:assert all(row['fallback'] for row in checked)
    else:assert all(row['original_Fock_residual']<=result.history[i]['inner'][0]['target']
                    for i,row in enumerate(checked) if not row['fallback'])


def test_invalid_compression_configuration_is_rejected():
    with pytest.raises(ValueError):solve_molecular_scf(model('He'),config=SCFConfig(exchange_compression_tolerance=float('nan')))
