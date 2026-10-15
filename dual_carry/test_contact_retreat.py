import unittest
import numpy as np
from contact_retreat import ContactRetreat


class RetreatTest(unittest.TestCase):
    def test_detection_hysteresis_and_distance_limit(self):
        c={'time_s':0.,'robots':[{'position_m':[-.6,0,.7]},{'position_m':[.6,0,.7]}],
           'object':{'position_m':[0,0,.4]},'contacts':{'pairs':[]}}
        r=ContactRetreat()
        c['contacts']['pairs']=[{'robot_index':0,'hand_index':0,'other_geom':'r1/hand','force_on_box_N':[0,0,20]}]
        r.update(c,True,.1);self.assertFalse(r.active.any())
        c['contacts']['pairs']=[{'robot_index':1,'hand_index':None,'other_geom':'r2/right_knee_link','force_on_box_N':[2,0,0]}]
        r.update(c,False,.1);self.assertFalse(r.active.any())
        r.update(c,True,.1);self.assertEqual(r.active.tolist(),[False,True])
        np.testing.assert_allclose(r.direction[1],[1,0])
        c['time_s']=.2;c['contacts']['pairs']=[]
        r.update(c,True,.1);self.assertTrue(r.active[1])
        c['time_s']=.5;r.update(c,True,.1);self.assertFalse(r.active.any())
        c['contacts']['pairs']=[{'robot_index':1,'hand_index':None,'other_geom':'r2/pelvis','force_on_box_N':[2,0,0]}]
        for j in range(100):
            c['time_s']+=.1;r.update(c,True,.1)
        self.assertLessEqual(np.linalg.norm(r.offset[1]),.16000001)
        c['robots'][1]['position_m'][0]=.78;r.update(c,True,.1)
        self.assertTrue(r.limited[1])


if __name__=='__main__':unittest.main()
