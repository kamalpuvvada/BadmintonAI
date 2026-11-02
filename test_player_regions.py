import unittest
import numpy as np
from player_regions import parse_polygon, polygon_mask, validate_polygon
from player_tracking import select_person_detections, annotate_detector_frame
from PIL import Image
from test_player_tracking import FakeDetections


class RegionTests(unittest.TestCase):
    def test_tapered_boundary_keeps_near_left_player_and_rejects_far_bystander(self):
        polygon = parse_polygon('0,0.61;0.42,0.53;1,0.53;1,1;0,1')
        points = [[.1,.65],[.2,.55],[.7,.56],[0,.61],[.1,.51]]
        self.assertEqual(polygon_mask(points, polygon).tolist(), [True,False,True,True,False])
        self.assertEqual(polygon_mask(points, polygon[::-1]).tolist(), [True,False,True,True,False])
        people=FakeDetections([[5,50,15,65],[15,40,25,55],[65,40,75,56]], [1,1,1],[.9,.9,.9])
        selected=select_person_detections(people,(100,100),None,None,court_polygon=polygon)
        self.assertEqual(len(selected),2)

    def test_degenerate_crossed_concave_nonfinite_regions_fail(self):
        for value in ('0,0;1,1;0,1;1,0','0,0;1,0;.2,.2;1,1;0,1',
                      '0,0;0,0;1,1','0,0;0.5,0.5;1,1','0,0;1,0;nan,1','0,0;2,0;1,1','0,0;1,0'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_polygon(value)
        self.assertIsNone(parse_polygon(None))

    def test_empty_points_and_invalid_shape(self):
        polygon=validate_polygon([[0,0],[1,0],[0,1]])
        self.assertEqual(polygon_mask([],polygon).tolist(),[])
        with self.assertRaises(ValueError):
            validate_polygon([[0,0,0],[1,0,0],[0,1,0]])

    def test_diagnostics_show_the_exact_inclusion_polygon(self):
        people=FakeDetections([],[],[])
        polygon=parse_polygon('0.1,0.1;0.9,0.1;0.9,0.9;0.1,0.9')
        image=annotate_detector_frame(Image.new('RGB',(100,100),'black'),people,people,None,None,polygon)
        self.assertEqual(image.getpixel((50,10)),(255,224,98))
