import numpy as np
from level_speed_coordination import level_rotation,allocate_vertical

def test_level_and_moment_direction():
    theta=.20;c,s=np.cos(theta),np.sin(theta)
    R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    desired=level_rotation(R)
    np.testing.assert_allclose(desired[:,2],[0,0,1],atol=1e-12)
    assert np.linalg.det(desired)>.999
    local=np.array([[-.2,.18,0],[-.2,-.18,0],[.2,-.18,0],[.2,.18,0]])
    current=local@R.T;level=local@desired.T
    assert np.all(level[2:,2]>current[2:,2])  # Raise the low (+X) side.
    moment=3*np.cross(R[:,2],[0,0,1.])
    f,residual=allocate_vertical(current,np.zeros(3),np.ones((2,2),bool),13.734,moment)
    assert f[1].sum()>f[0].sum()
    actual=np.cross(current,np.c_[np.zeros((4,2)),f.reshape(-1)]).sum(0)
    assert actual[1]<0 and actual[1]*moment[1]>0
    assert abs(f.sum()-13.734)<.1
    assert np.linalg.norm(residual[1:])<.1

def test_contact_loss_and_limits():
    pts=np.array([[-.2,.18,0],[-.2,-.18,0],[.2,-.18,0],[.2,.18,0]])
    contacts=np.array([[True,False],[True,False]])
    f,residual=allocate_vertical(pts,np.zeros(3),contacts,13.734,np.array([0.,-.3,0.]))
    assert np.all(f[~contacts]==0) and np.all(f>=0) and np.all(f<=12)
    f,residual=allocate_vertical(pts,np.zeros(3),np.zeros((2,2),bool),13.734,np.zeros(3))
    assert np.all(f==0) and abs(residual[0]+13.734)<1e-10
    f,residual=allocate_vertical(pts,np.zeros(3),np.ones((2,2),bool),100,np.array([50.,-50.,0.]))
    assert np.all(f>=0) and np.all(f<=12) and np.isfinite(residual).all()
    assert np.linalg.norm(residual)>1  # Infeasible wrench must be reported.

if __name__=='__main__':
    test_level_and_moment_direction();test_contact_loss_and_limits()
    print('PASS: level target sign, corrective moment sign, contact masks, force bounds, infeasible residual')
