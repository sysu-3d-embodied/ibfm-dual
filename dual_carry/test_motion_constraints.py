"""Focused checks for reverse-motion exclusion and coordinate handling."""
import unittest
import torch
from motion_constraints import directional_candidates


class MotionConstraintsTest(unittest.TestCase):
    def test_high_other_rewards_cannot_restore_reverse_sample(self):
        velocity=torch.tensor([[0.,-.18,0.],[0.,.10,0.],[0.,0.,0.]])
        valid,_=directional_candidates(velocity,torch.tensor([0.,.1,0.]),'transport')
        # Even an arbitrarily large posture/grasp score stays excluded.
        prob=torch.softmax(torch.tensor([1000.,.5,.3]).masked_fill(~valid,-torch.inf),dim=0)
        self.assertEqual(prob[0].item(),0.)
        self.assertEqual(valid.tolist(),[False,True,True])

    def test_opposite_robot_frames_have_same_world_filter(self):
        world=torch.tensor([[0.,-.18,0.],[0.,.1,0.],[.1,.02,0.]])
        target=torch.tensor([0.,.1,0.])
        # Also include roll: projecting local XY before rotating is incorrect.
        rotation=torch.tensor([[-1.,0.,0.],[0.,-.8,.6],[0.,.6,.8]])
        local=world@rotation;local_target=target@rotation
        first,_=directional_candidates(world,target,'transport')
        second,_=directional_candidates(local@rotation.T,local_target@rotation.T,'transport')
        self.assertTrue(torch.equal(first,second))

    def test_zero_target_has_no_direction_filter(self):
        valid,meta=directional_candidates(torch.tensor([[0.,-.2,0.]]),torch.zeros(3),'transport')
        self.assertTrue(valid.all());self.assertFalse(meta['direction_filter_active'])

    def test_lift_is_not_filtered_as_transport(self):
        valid,_=directional_candidates(torch.tensor([[0.,-.2,.1]]),torch.tensor([0.,.1,.1]),'lift')
        self.assertTrue(valid.all())

    def test_lower_rejects_upward_but_accepts_descent(self):
        valid,_=directional_candidates(torch.tensor([[0.,0.,.1],[0.,0.,-.05]]),torch.tensor([0.,0.,-.05]),'lower')
        self.assertEqual(valid.tolist(),[False,True])

    def test_empty_support_is_reported_not_relaxed(self):
        valid,_=directional_candidates(torch.tensor([[0.,-.2,0.],[0.,-.1,0.]]),torch.tensor([0.,.1,0.]),'transport')
        self.assertFalse(valid.any())


if __name__=='__main__':unittest.main()
