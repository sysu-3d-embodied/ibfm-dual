import unittest
import numpy as np
import torch
from goal_steering import goal_command,supported_weights,steering_weight,wrap


class GoalSteeringTest(unittest.TestCase):
    def test_target_from_current_position(self):
        v,yaw,d=goal_command([1,1],[5,0],.1,0,.1)
        self.assertGreater(v[0],0);self.assertLess(v[1],0)
        self.assertAlmostEqual(v[1]/v[0],-.25)
        self.assertAlmostEqual(np.linalg.norm(v),.1)

    def test_heading_error_does_not_stop_motion(self):
        a=goal_command([0,0],[5,0],.1,0,.1)[0]
        b=goal_command([0,0],[5,0],.1,np.pi/2,.1)[0]
        np.testing.assert_allclose(a,b)

    def test_opposed_roles_and_turn_sign(self):
        v,heading,_=goal_command([0,0],[5,0],.1,0,.1)
        self.assertGreater(v@np.array([np.cos(heading),np.sin(heading)]),0)
        self.assertLess(v@np.array([np.cos(heading+np.pi),np.sin(heading+np.pi)]),0)
        self.assertEqual(steering_weight(.2,0)[0],1)
        self.assertEqual(steering_weight(-.2,0)[0],-1)
        self.assertAlmostEqual(float(wrap(-np.pi+.1-(np.pi-.1))),.2)

    def test_arrival_and_overshoot(self):
        far=goal_command([0,0],[1,0],.1,0,.1)[0]
        near=goal_command([.99,0],[1,0],.1,0,.1)[0]
        self.assertLess(np.linalg.norm(near),np.linalg.norm(far))
        self.assertLess(goal_command([1.1,0],[1,0],.1,0,.1)[0][0],0)

    def test_support_and_diversity(self):
        score=torch.ones(100)*.5;score[0]=1.;score[-1]=20
        valid=torch.ones(100,dtype=torch.bool);valid[-1]=False
        w,d=supported_weights(score,valid,40.,16.)
        self.assertEqual(float(w[-1]),0)
        self.assertGreaterEqual(d['effective_samples'],16)
        empty,d=supported_weights(score,torch.zeros_like(valid))
        self.assertTrue(d['no_feasible_candidates']);self.assertEqual(float(empty.sum()),0)


if __name__=='__main__':unittest.main()
