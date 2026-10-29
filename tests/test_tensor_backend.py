"""Independent tensor/Galerkin checks and shared nonlinear solver contracts."""
import numpy as np
import pytest
from scipy.linalg import solve
from scipy.sparse.linalg import LinearOperator

from hf_fas.tensor import TensorMesh, TensorQ1Space, TensorCoulomb, TensorCoulombConfig
from hf_fas.tensor.coulomb import voxel_kernel_samples, voxel_kernel_reference, prism_integral
from hf_fas.fem import (MolecularFEMModel, MolecularResidual, regression_molecules,
                       SCFConfig, solve_molecular_scf, FEMHierarchy, FASConfig)
from hf_fas.fem.coarse import ReducedCoordinates, CoarseConfig, linear_newton_step
from hf_fas.experiments.fem_paper import variational_qualification, precision_endpoint_check, gaussian_control


@pytest.fixture(scope='module',params=['fft','vgf'])
def tensor_context(request):
    space=TensorQ1Space(TensorMesh(4,4.))
    return space,TensorCoulomb(space,TensorCoulombConfig(method=request.param))


def test_consistent_tensor_mass_and_elliptic_inverses(tensor_context):
    space,_=tensor_context; rng=np.random.default_rng(10)
    assert isinstance(space.mass,LinearOperator)
    assert np.count_nonzero(space.mass.toarray()-np.diag(space.mass.diagonal()))>0
    for inverse in (space.mass_inverse,space.pair_mass_inverse,space.auxiliary_inverse,space.kinetic_inverse):
        rhs=rng.normal(size=(inverse.matrix.shape[0],3))
        expected=solve(inverse.matrix.toarray(),rhs,assume_a='pos')
        np.testing.assert_allclose(inverse.solve(rhs),expected,atol=2e-12,rtol=2e-12)
        empty=np.zeros((inverse.matrix.shape[0],0,0))
        assert inverse.solve(empty).shape==empty.shape
    lower,upper=space.mass_spectral_bounds
    eigen=np.linalg.eigvalsh(space.mass.toarray())
    assert lower<=eigen.min() and upper>=eigen.max()
    root=space.mass_cholesky
    np.testing.assert_allclose(root.toarray() @ root.toarray().T,space.mass.toarray(),atol=2e-14)
    rhs=rng.normal(size=(space.n_orbital,3))
    np.testing.assert_allclose(root.solve(rhs),solve(root.toarray(),rhs),atol=2e-13)
    np.testing.assert_allclose(root.solve(rhs,transpose=True),solve(root.toarray().T,rhs),atol=2e-13)


def test_L2_product_projection_charge_and_weak_adjoint(tensor_context):
    space,_=tensor_context; rng=np.random.default_rng(11)
    p=rng.normal(size=(space.n_orbital,2)); v=rng.normal(size=(space.n_orbital,3))
    source=space.source_products(p,v); load=rng.normal(size=source.shape)
    assert np.sum(source*load)==pytest.approx(np.sum(v*space.source_adjoint(p,load)),rel=2e-13)
    density=space.sources(p)
    charge=np.ones(space.n_pair) @ (space.pair_mass @ density.reshape(space.n_pair,-1))
    np.testing.assert_allclose(charge.reshape(2,2),p.T @ (space.mass @ p),atol=2e-12)
    nodal=space.evaluation @ p
    assert np.linalg.norm(density-np.einsum('ai,aj->aij',nodal,nodal))>1e-3
    w=rng.normal(size=space.n_pair)
    np.testing.assert_allclose(space.potential_matrix(w) @ p,space.multiply_load(w,p),atol=2e-12)


def test_integrated_self_kernel_independent_spatial_Duffy():
    nodes,weights=np.polynomial.legendre.leggauss(36);q,w=(nodes+1)/2,weights/2
    v,z=np.meshgrid(q,q,indexing='ij')
    radial=.5-(1+v+v*z)/3+(v+v*z+v*v*z)/4-v*v*z/5
    integral=48*np.sum(w[:,None]*w[None,:]*v*radial/np.sqrt(1+v*v+(v*z)**2))
    assert voxel_kernel_samples([[0,0,0]],256)[0]==pytest.approx(integral,rel=2e-13)
    np.testing.assert_allclose(prism_integral(np.full((1,3),-.5),np.full((1,3),.5)),[2.38007736397955],rtol=2e-13)
    offsets=[[1,0,0],[2,3,4],[127,127,127]]
    values=voxel_kernel_samples(offsets,256)
    np.testing.assert_allclose(values,[voxel_kernel_reference(x) for x in offsets],rtol=2e-12)


def test_zero_padded_FFT_matches_direct_voxel_Galerkin_and_is_positive():
    space=TensorQ1Space(TensorMesh(2,2.));c=TensorCoulomb(space,TensorCoulombConfig(method="fft",subdivisions=2))
    indices=np.stack(np.meshgrid(*[np.arange(c.side)]*3,indexing='ij'),axis=-1).reshape(-1,3)
    offsets=np.abs(indices[:,None]-indices[None,:]).reshape(-1,3)
    kernel=voxel_kernel_samples(offsets,256).reshape(len(indices),len(indices))*c.voxel_spacing**5
    b=c.evaluation.toarray(); expected=b.T @ kernel @ b
    actual=c.integral_action(np.eye(space.n_pair))
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
    assert np.linalg.eigvalsh((actual+actual.T)/2).min()>-1e-12
    np.testing.assert_allclose(actual,actual.T,atol=2e-13)


def test_FFT_kernel_checked_reuse_and_no_FMM_dependency(tensor_context,tmp_path,monkeypatch):
    space,c=tensor_context
    a=TensorCoulomb(space,c.config,kernel_cache=tmp_path)
    def no_build(*args,**kwargs):raise AssertionError('Warm kernel must not be rebuilt')
    monkeypatch.setattr(TensorCoulomb,'_build_kernel',no_build)
    b=TensorCoulomb(space,c.config,kernel_cache=tmp_path)
    assert a.fingerprint==b.fingerprint and b.cache_info['status']=='hit'
    np.testing.assert_array_equal(a.kernel_fft,b.kernel_fft)


@pytest.mark.parametrize('system',['He','LiH','BH','H2O'])
def test_four_molecule_energy_response_SCF_and_strict_endpoint(tensor_context,system):
    space,c=tensor_context; model=MolecularFEMModel(space,regression_molecules()[system],c)
    assert variational_qualification(model)['qualified']
    result=solve_molecular_scf(model,config=SCFConfig(eigensolver='dense',tolerance=1e-9))
    assert result.converged, result.history[-1]
    assert precision_endpoint_check(model,result.state)['qualified']
    p=result.state.orbitals[0]
    np.testing.assert_allclose(model.fock_load(result.state,0,p),model.action_load(result.state.orbitals,result.state.pairs)[0],atol=2e-11)
    assert model.electron_count(result.state)==pytest.approx(model.molecule.electrons,abs=2e-12)


def test_tensor_forced_tau_mass_adjoints_and_fixed_point():
    spaces=[TensorQ1Space(TensorMesh(m,4.)) for m in (3,6)]
    models=[MolecularFEMModel(s,regression_molecules()['LiH'],TensorCoulomb(s)) for s in spaces]
    hierarchy=FEMHierarchy(models);transfer=hierarchy.transfers[0];coarse,fine=models
    state=fine.initial_state();forcing=fine.operator(state)
    rng=np.random.default_rng(12)
    a=rng.normal(size=(coarse.space.n_orbital,2));b=rng.normal(size=(fine.space.n_orbital,2))
    assert np.sum((transfer.orbital_prolongation@a)*(fine.space.mass@b))==pytest.approx(np.sum(a*(coarse.space.mass@transfer.restrict_orbitals(b))),rel=2e-13)
    restricted,rhs=transfer.coarse_equation(state,forcing)
    assert coarse.residual(restricted,rhs).norm(coarse.space)<1e-11
    result=hierarchy.cycle(1,state.copy(),forcing,FASConfig(pre_sweeps=0,post_sweeps=0))
    assert fine.residual(result,forcing).norm(fine.space)<1e-10
    q=np.linalg.qr(rng.normal(size=(2,2)))[0]
    rotated=fine.rebuild((state.orbitals[0]@q,),forcing.rotate((q,)))
    expected=state.rotate((q,))
    np.testing.assert_allclose(rotated.pairs[0],expected.pairs[0],atol=2e-12)


def test_large_tensor_Riesz_Krylov_branch():
    space=TensorQ1Space(TensorMesh(10,4.))
    model=MolecularFEMModel(space,regression_molecules()['He'],TensorCoulomb(space))
    state=model.initial_state();coordinates=ReducedCoordinates(model)
    assert isinstance(coordinates.root,LinearOperator)
    forcing=MolecularResidual.zeros(state);rhs=-coordinates.residual(state,forcing)
    _,record=linear_newton_step(model,state,coordinates,rhs,CoarseConfig(krylov_tolerance=1e-8))
    assert record['converged'],record


def test_tensor_Gaussian_physical_normalization_and_tail():
    space=TensorQ1Space(TensorMesh(8,4.))
    report=gaussian_control(space,TensorCoulomb(space),4.)
    assert report['relative_projected_potential_error']<.03
    assert report['maximum_probe_error']<.07
    assert abs(report['represented_charge']+report['exact_outside_source_mass']-1)<1e-10




def test_large_trial_Gram_roundoff_is_symmetric_and_invalid_targets_stay_errors(tensor_context):
    space,c=tensor_context;model=MolecularFEMModel(space,regression_molecules()['LiH'],c)
    rng=np.random.default_rng(20261018)
    trial=1e7*rng.normal(size=(space.n_orbital,2))
    target=np.array([[1.4,.2],[.2,.8]])
    frame=model.retract(trial,target)
    np.testing.assert_allclose(model.gram(frame),target,atol=2e-12)
    target[0,1]+=.01
    with pytest.raises(ValueError,match='symmetric'):
        model.retract(trial,target)


def test_joint_Newton_does_not_stop_before_charge_constraint(tensor_context):
    from hf_fas.fem.coarse import solve_forced_coarse
    space,c=tensor_context;model=MolecularFEMModel(space,regression_molecules()['He'],c)
    physical=solve_molecular_scf(model,config=SCFConfig(eigensolver='dense',tolerance=1e-12)).state
    state=model.rebuild(tuple(p*np.sqrt(1+7e-10) for p in physical.orbitals),retract=False)
    state.multipliers=physical.multipliers
    assert model.residual(state).norm(space)<1e-8
    assert abs(model.electron_count(state)-2)>1e-9
    result,report=solve_forced_coarse(model,state,MolecularResidual.zeros(state),
        CoarseConfig(tolerance=1e-8,gs_fallback=False),rebuild_initial=False)
    assert report['converged'] and len(report['history'])>1
    assert abs(model.electron_count(result)-2)<1e-9
