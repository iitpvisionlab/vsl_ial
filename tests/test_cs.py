import unittest
from vsl_ial.cs import (
    convert,
    XYZ,
    sRGB,
    linRGB,
    LMS,
    JzAzBz,
    D65,
    CS,
    Oklab,
    ICaCb,
    ICtCp,
    CIELUV,
)
from vsl_ial.cs.prolab import ProLab
from vsl_ial.cs.cam import Average
import numpy as np
import numpy.typing as npt


class TestCaseCSBase:
    # https://en.wikipedia.org/wiki/HSL_and_HSV
    wikipedia_sRGB_colors = (
        (1.000, 1.000, 1.000),
        (0.500, 0.500, 0.500),
        (0.000, 0.000, 0.000),
        (1.000, 0.000, 0.000),
        (0.750, 0.750, 0.000),
        (0.000, 0.500, 0.000),
        (0.500, 1.000, 1.000),
        (0.500, 0.500, 1.000),
        (0.750, 0.250, 0.750),
        (0.628, 0.643, 0.142),
        (0.255, 0.104, 0.918),
        (0.116, 0.675, 0.255),
        (0.941, 0.785, 0.053),
        (0.704, 0.187, 0.897),
        (0.931, 0.463, 0.316),
        (0.998, 0.974, 0.532),
        (0.099, 0.795, 0.591),
        (0.211, 0.149, 0.597),
        (0.495, 0.493, 0.721),
    )

    def setUp(self):
        np.seterr(all="raise")

    def _test(
        self,
        src: CS,
        dst: CS,
        color: npt.ArrayLike,
        ref: npt.ArrayLike,
    ) -> None:
        raise NotImplementedError

    def test_XYZ_to_sRGB(self):
        self._test(
            XYZ(),
            sRGB(),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.4, 0.2, 0.6],
        )

    def test_XYZ_to_linRGB(self):
        self._test(
            XYZ(),
            linRGB(),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.13287, 0.0331, 0.31855],
        )

    def test_XYZ_to_LMS(self):
        self._test(
            XYZ(),
            LMS(),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.0777, 0.0734, 0.284],
        )

    def test_LMS_to_XYZ(self):
        self._test(
            LMS(),
            XYZ(),
            color=[0.0777, 0.0734, 0.284],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_sRGB_to_XYZ(self):
        self._test(
            sRGB(),
            XYZ(),
            color=[0.4, 0.2, 0.6],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_sRGB_to_HLS(self):
        from vsl_ial.cs import HLS
        from colorsys import rgb_to_hls

        for color in self.wikipedia_sRGB_colors:
            self._test(
                sRGB(),
                HLS(),
                color=color,
                ref=rgb_to_hls(*color),
            )

    def test_HSL_to_sRGB(self):
        from vsl_ial.cs import HLS
        from colorsys import rgb_to_hls

        for color in self.wikipedia_sRGB_colors:
            self._test(
                HLS(),
                sRGB(),
                color=rgb_to_hls(*color),
                ref=color,
            )

    def test_sRGB_to_HSV(self):
        from vsl_ial.cs import HSV
        from colorsys import rgb_to_hsv

        for color in self.wikipedia_sRGB_colors:
            self._test(
                sRGB(),
                HSV(),
                color=color,
                ref=rgb_to_hsv(*color),
            )

    def test_HSV_to_sRGB(self):
        from vsl_ial.cs import HSV
        from colorsys import rgb_to_hsv

        for color in self.wikipedia_sRGB_colors:
            self._test(
                HSV(),
                sRGB(),
                color=rgb_to_hsv(*color),
                ref=color,
            )

    def test_XYZ_to_Jzazbz(self):
        self._test(
            XYZ(),
            JzAzBz(),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.00391332, 0.00331202, -0.00905872],
        )

    def test_Jzazbz_to_XYZ(self):
        self._test(
            JzAzBz(),
            XYZ(),
            color=[0.00391332, 0.00331202, -0.00905872],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_Oklab(self):
        self._test(
            XYZ(),
            Oklab(),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.44027, 0.08818, -0.13394],
        )

    def test_Oklab_to_XYZ(self):
        self._test(
            Oklab(),
            XYZ(),
            color=[0.44027, 0.08818, -0.13394],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_ICaCb(self):
        self._test(
            XYZ(),
            ICaCb(),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.0561, 0.0172, -0.0513],
        )

    def test_ICaCb_to_XYZ(self):
        self._test(
            ICaCb(),
            XYZ(),
            color=[0.0561, 0.0172, -0.0513],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_ICtCp(self):
        self._test(
            XYZ(),
            ICtCp(),
            color=[0.12412, 0.07493, 0.3093],
            ref=[527.4813, 82.1905, -66.3334],
        )

    def test_ICtCp_to_XYZ(self):
        self._test(
            ICtCp(),
            XYZ(),
            color=[527.4813, 82.1905, -66.3334],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_CIELUV(self):
        self._test(
            XYZ(),
            CIELUV(illuminant_xyz=D65),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.329039, 0.129716, -0.677639],
        )

    def test_CIELUV_to_XYZ(self):
        self._test(
            CIELUV(illuminant_xyz=D65),
            XYZ(),
            color=[0.329039, 0.129716, -0.677639],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_PROLAB(self):
        self._test(
            XYZ(),
            ProLab(illuminant_xyz=D65),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.5037, 0.1595, -0.2585],
        )

    def test_PROLAB_to_XYZ(self):
        self._test(
            ProLab(illuminant_xyz=D65),
            XYZ(),
            color=[0.5037, 0.1595, -0.2585],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_CAM16(self):
        # import colour
        # res = colour.XYZ_to_CIECAM16(
        #     XYZ=np.asarray([0.12412, 0.07493, 0.3093]) * 100.0,
        #     XYZ_w=np.asarray(D65) * 100.0, L_A=60.0, Y_b=20.0)
        # print("res", res.J, res.M, res.h)

        from vsl_ial.cs.cam import CAM16

        self._test(
            XYZ(),
            CAM16(illuminant_xyz=D65, L_A=0.6, Y_b=0.2, surround=Average),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.254849690045, 0.507680464129, np.radians(310.433972897)],
        )

    def test_CAM16_to_XYZ(self):
        # import colour
        # res = colour.XYZ_to_CIECAM16(
        #     XYZ=np.asarray([0.12412, 0.07493, 0.3093]) * 100.0,
        #     XYZ_w=np.asarray(D65) * 100.0, L_A=60.0, Y_b=20.0)
        # print("res", res.J, res.M, res.h)

        from vsl_ial.cs.cam import CAM16

        self._test(
            CAM16(illuminant_xyz=D65, L_A=0.6, Y_b=0.2, surround=Average),
            XYZ(),
            color=[0.254849690045, 0.507680464129, np.radians(310.433972897)],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_CAM16UCS(self):
        from vsl_ial.cs.cam import CAM16UCS

        self._test(
            XYZ(),
            CAM16UCS(illuminant_xyz=D65, L_A=0.6, Y_b=0.2, surround=Average),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.3676564758394433, 0.21873792289758843, -0.2567076432448853],
        )

    def test_CAM16UCS_to_XYZ(self):
        from vsl_ial.cs.cam import CAM16UCS

        self._test(
            CAM16UCS(illuminant_xyz=D65, L_A=0.6, Y_b=0.2, surround=Average),
            XYZ(),
            color=[
                0.3676564758394433,
                0.21873792289758843,
                -0.2567076432448853,
            ],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_CIELAB(self):
        from vsl_ial.cs.cielab import CIELAB

        self._test(
            XYZ(),
            CIELAB(illuminant_xyz=D65),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.329039, 0.428786, -0.47156],
        )
        self._test(
            XYZ(),
            CIELAB(illuminant_xyz=D65),
            color=[0.0087, 0.0085, 0.0135],
            ref=[0.076780185185185, 0.02531210019708, -0.054659059902865],
        )

    def test_CIELAB_to_XYZ(self):
        from vsl_ial.cs.cielab import CIELAB

        self._test(
            CIELAB(illuminant_xyz=D65),
            XYZ(),
            color=[0.329039 * 1.0, 0.428786 * 1.0, -0.47156 * 1.0],
            ref=[0.12412, 0.07493, 0.3093],
        )
        self._test(
            CIELAB(illuminant_xyz=D65),
            XYZ(),
            color=[0.076780185185185, 0.02531210019708, -0.054659059902865],
            ref=[0.0087, 0.0085, 0.0135],
        )

    def test_XYZ_to_CAM02(self):
        from vsl_ial.cs.cam import CAM02

        self._test(
            XYZ(),
            CAM02(illuminant_xyz=D65, L_A=0.6, Y_b=0.2, surround=Average),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.246880325931, 0.484522681528, np.radians(299.40091582)],
        )

    def test_CAM02_to_XYZ(self):
        from vsl_ial.cs.cam import CAM02

        self._test(
            CAM02(illuminant_xyz=D65, L_A=0.6, Y_b=0.2, surround=Average),
            XYZ(),
            color=[0.246880325931, 0.484522681528, np.radians(299.40091582)],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_xyY(self):
        from vsl_ial.cs.ciexyy import CIExyY

        self._test(
            XYZ(),
            CIExyY(illuminant_xyz=D65),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.2441624864758532, 0.1473984459525917, 0.07493],
        )

    def test_xyY_to_XYZ(self):
        from vsl_ial.cs.ciexyy import CIExyY

        self._test(
            CIExyY(illuminant_xyz=D65),
            XYZ(),
            color=[0.2441624864758532, 0.1473984459525917, 0.07493],
            ref=[0.12412, 0.07493, 0.3093],
        )

    def test_XYZ_to_Opponent(self):
        from vsl_ial.cs.opponent import Opponent

        self._test(
            XYZ(),
            Opponent(),
            color=[0.12412, 0.07493, 0.3093],
            ref=[0.05548398, -0.05781628, 0.12142492],
        )

    def test_Opponent_to_XYZ(self):
        from vsl_ial.cs.opponent import Opponent

        self._test(
            Opponent(),
            XYZ(),
            color=[0.05548398, -0.05781628, 0.12142492],
            ref=[0.12412, 0.07493, 0.3093],
        )


class TestCaseCS1D(TestCaseCSBase, unittest.TestCase):
    def _test(
        self,
        src: CS,
        dst: CS,
        color: npt.ArrayLike,
        ref: npt.ArrayLike,
    ):
        res_1d = convert(src, dst, color=color)
        np.testing.assert_almost_equal(res_1d, ref, decimal=4)


class TestCaseCS2D(TestCaseCSBase, unittest.TestCase):
    def _test(
        self,
        src: CS,
        dst: CS,
        color: npt.ArrayLike,
        ref: npt.ArrayLike,
    ):
        res_2d = convert(src, dst, color=[color, color])
        np.testing.assert_almost_equal(res_2d, [ref, ref], decimal=4)


class TestCaseCS3D(TestCaseCSBase, unittest.TestCase):
    def _test(
        self,
        src: CS,
        dst: CS,
        color: npt.ArrayLike,
        ref: npt.ArrayLike,
    ):
        res_3d = convert(
            src, dst, color=[[color, color, color], [color, color, color]]
        )
        np.testing.assert_almost_equal(
            res_3d, [[ref, ref, ref], [ref, ref, ref]], decimal=4
        )
