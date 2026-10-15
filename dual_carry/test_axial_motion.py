import unittest
import torch
from axial_motion import axial_candidates,heading_error_from_relative_rotation


class AxialTest(unittest.TestCase):
    def sample(self,yaw):
        yaw=torch.tensor(yaw);c,s=torch.cos(yaw),torch.sin(yaw)
        S=torch.zeros((len(yaw),98));S[:,67]=c;S[:,68]=s;S[:,72]=1
        return S

    def test_relative_heading(self):
        S=self.sample([0.,torch.pi/2,-torch.pi/2]);target=torch.tensor([1.,0,0,0,0,1])
        error=heading_error_from_relative_rotation(S,target)
        self.assertTrue(torch.allclose(error,torch.tensor([0.,-torch.pi/2,torch.pi/2]),atol=1e-6))

    def test_roles_and_sideways_exclusion(self):
        S=self.sample([0.,0.,0.,torch.pi/2]);F=torch.zeros((4,6))
        F[:,0]=torch.tensor([.1,-.1,.1,.1]);F[2,1]=.2
        target=torch.tensor([1.,0,0,0,0,1])
        f,_=axial_candidates(S,F,target,1.);b,_=axial_candidates(S,F,target,-1.)
        self.assertEqual(f.tolist(),[True,False,False,False])
        self.assertEqual(b.tolist(),[False,True,False,False])

    def test_rotated_reference(self):
        S=self.sample([torch.pi/2]);target=S[0,67:73]
        self.assertTrue(torch.allclose(heading_error_from_relative_rotation(S,target),torch.zeros(1),atol=1e-6))


if __name__=='__main__':unittest.main()
