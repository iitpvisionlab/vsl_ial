import unittest
from typing import Callable
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
    whitepoints_cie1964,
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

    def _test_f(
        self,
        f: Callable[[npt.ArrayLike], npt.ArrayLike],
        color: npt.ArrayLike,
        ref: npt.ArrayLike,
    ) -> None:
        raise NotImplementedError

    def test_XYZ_to_sRGB(self):
        self._test(
            XYZ(),
            sRGB(),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.4, 0.2, 0.6],
            ref1=[0.1, 1.0, 0.0],
        )

    def test_XYZ_to_linRGB(self):
        self._test(
            XYZ(),
            linRGB(),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.13287, 0.0331, 0.31855],
            ref1=[0.010022864290, 1.0, 0.0],
        )

    def test_XYZ_to_LMS(self):
        self._test(
            XYZ(),
            LMS(),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.0777, 0.0734, 0.284],
            ref1=[0.642673281, 0.7594660777, 0.109622411],
        )

    def test_LMS_to_XYZ(self):
        self._test(
            LMS(),
            XYZ(),
            color0=[0.0777, 0.0734, 0.284],
            color1=[0.642673281, 0.7594660777, 0.109622411],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_sRGB_to_XYZ(self):
        self._test(
            sRGB(),
            XYZ(),
            color0=[0.4, 0.2, 0.6],
            color1=[0.1, 1.0, 0.0],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_sRGB_to_HLS(self):
        from vsl_ial.cs import HLS
        from colorsys import rgb_to_hls

        color1 = self.wikipedia_sRGB_colors[0]
        ref1 = rgb_to_hls(*color1)

        for color in self.wikipedia_sRGB_colors[1:]:
            self._test(
                sRGB(),
                HLS(),
                color0=color,
                color1=color1,
                ref0=rgb_to_hls(*color),
                ref1=ref1,
            )

    def test_HSL_to_sRGB(self):
        from vsl_ial.cs import HLS
        from colorsys import rgb_to_hls

        ref1 = self.wikipedia_sRGB_colors[0]
        color1 = rgb_to_hls(*ref1)

        for color in self.wikipedia_sRGB_colors[1:]:
            self._test(
                HLS(),
                sRGB(),
                color0=rgb_to_hls(*color),
                color1=color1,
                ref0=color,
                ref1=ref1,
            )

    def test_sRGB_to_HSV(self):
        from vsl_ial.cs import HSV
        from colorsys import rgb_to_hsv

        color1 = self.wikipedia_sRGB_colors[0]
        ref1 = rgb_to_hsv(*color1)

        for color in self.wikipedia_sRGB_colors[1:]:
            self._test(
                sRGB(),
                HSV(),
                color0=color,
                color1=color1,
                ref0=rgb_to_hsv(*color),
                ref1=ref1,
            )

    def test_HSV_to_sRGB(self):
        from vsl_ial.cs import HSV
        from colorsys import rgb_to_hsv

        ref1 = self.wikipedia_sRGB_colors[0]
        color1 = rgb_to_hsv(*ref1)
        for color in self.wikipedia_sRGB_colors[1:]:
            self._test(
                HSV(),
                sRGB(),
                color0=rgb_to_hsv(*color),
                color1=color1,
                ref0=color,
                ref1=ref1,
            )

    def test_XYZ_to_Jzazbz(self):
        self._test(
            XYZ(),
            JzAzBz(),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.00391332, 0.00331202, -0.00905872],
            ref1=[0.0124749172, -0.016089371215, 0.01661463483],
        )

    def test_Jzazbz_to_XYZ(self):
        self._test(
            JzAzBz(),
            XYZ(),
            color0=[0.00391332, 0.00331202, -0.00905872],
            color1=[0.0124749172, -0.016089371215, 0.01661463483],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_XYZ_to_Oklab(self):
        self._test(
            XYZ(),
            Oklab(),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.44027, 0.08818, -0.13394],
            ref1=[0.86759324, -0.2317104079, 0.17963248724],
        )

    def test_Oklab_to_XYZ(self):
        self._test(
            Oklab(),
            XYZ(),
            color0=[0.44027, 0.08818, -0.13394],
            color1=[0.86759324, -0.2317104079, 0.17963248724],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_XYZ_to_ICaCb(self):
        self._test(
            XYZ(),
            ICaCb(),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.0561, 0.0172, -0.0513],
            ref1=[0.13163832448, -0.046759702599, 0.07293014],
        )

    def test_ICaCb_to_XYZ(self):
        self._test(
            ICaCb(),
            XYZ(),
            color0=[0.0561, 0.0172, -0.0513],
            color1=[0.13163832448, -0.046759702599, 0.07293014],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_XYZ_to_ICtCp(self):
        self._test(
            XYZ(),
            ICtCp(),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[527.4813, 82.1905, -66.3334],
            ref1=[690.2125025, -111.3910404, -127.348758],
        )

    def test_ICtCp_to_XYZ(self):
        self._test(
            ICtCp(),
            XYZ(),
            color0=[527.4813, 82.1905, -66.3334],
            color1=[690.2125025, -111.3910404, -127.348758],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_XYZ_to_CIELUV(self):
        self._test(
            XYZ(),
            CIELUV(illuminant_xyz=D65),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.329039, 0.129716, -0.677639],
            ref1=[0.8783768406, -0.819863499, 1.0738000689],
        )

    def test_CIELUV_to_XYZ(self):
        self._test(
            CIELUV(illuminant_xyz=D65),
            XYZ(),
            color0=[0.329039, 0.129716, -0.677639],
            color1=[0.8783768406, -0.819863499, 1.0738000689],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_XYZ_to_PROLAB(self):
        self._test(
            XYZ(),
            ProLab(illuminant_xyz=D65),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.5037, 0.1595, -0.2585],
            ref1=[0.9333917145, -0.45817234969, 0.30993073925],
        )

    def test_PROLAB_to_XYZ(self):
        self._test(
            ProLab(illuminant_xyz=D65),
            XYZ(),
            color0=[0.5037, 0.1595, -0.2585],
            color1=[0.9333917145, -0.45817234969, 0.30993073925],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
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
            CAM16(illuminant_xyz=D65, L_A=60, Y_b=20, surround=Average),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.254849690045, 0.507680464129, np.radians(310.433972897)],
            ref1=[0.7942661024, 0.9684004744, 2.470480293],
        )

    def test_CAM16_to_XYZ(self):
        # import colour
        # res = colour.XYZ_to_CIECAM16(
        #     XYZ=np.asarray([0.12412, 0.07493, 0.3093]) * 100.0,
        #     XYZ_w=np.asarray(D65) * 100.0, L_A=60.0, Y_b=20.0)
        # print("res", res.J, res.M, res.h)

        from vsl_ial.cs.cam import CAM16

        self._test(
            CAM16(illuminant_xyz=D65, L_A=60, Y_b=20, surround=Average),
            XYZ(),
            color0=[0.254849690045, 0.507680464129, np.radians(310.433972897)],
            color1=[0.7942661024, 0.9684004744, 2.470480293],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_XYZ_to_CAM16UCS(self):
        from vsl_ial.cs.cam import CAM16UCS

        self._test(
            XYZ(),
            CAM16UCS(illuminant_xyz=D65, L_A=60, Y_b=20, surround=Average),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.367656475839, 0.2187379228975, -0.256707643244],
            ref1=[0.8677791049, -0.4003696108, 0.31792002506],
        )

    def test_CAM16UCS_to_XYZ(self):
        from vsl_ial.cs.cam import CAM16UCS

        self._test(
            CAM16UCS(illuminant_xyz=D65, L_A=60, Y_b=20, surround=Average),
            XYZ(),
            color0=[0.367656475839, 0.2187379228975, -0.256707643244],
            color1=[0.8677791049, -0.4003696108, 0.31792002506],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_XYZ_to_CIELAB(self):
        from vsl_ial.cs.cielab import CIELAB

        self._test(
            XYZ(),
            CIELAB(illuminant_xyz=D65),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.0087, 0.0085, 0.0135],
            ref0=[0.329039, 0.428786, -0.47156],
            ref1=[0.076780185185185, 0.02531210019708, -0.054659059902865],
        )

    def test_CIELAB_to_XYZ(self):
        from vsl_ial.cs.cielab import CIELAB

        self._test(
            CIELAB(illuminant_xyz=D65),
            XYZ(),
            color0=[0.329039 * 1.0, 0.428786 * 1.0, -0.47156 * 1.0],
            color1=[0.076780185185185, 0.02531210019708, -0.054659059902865],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.0087, 0.0085, 0.0135],
        )

    def test_XYZ_to_CAM02(self):
        from vsl_ial.cs.cam import CAM02

        self._test(
            XYZ(),
            CAM02(illuminant_xyz=D65, L_A=60, Y_b=20, surround=Average),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.246880325931, 0.484522681528, np.radians(299.40091582)],
            ref1=[0.8006454338, 0.9505764278, 2.3723678665],
        )

    def test_CAM02_to_XYZ(self):
        from vsl_ial.cs.cam import CAM02

        self._test(
            CAM02(illuminant_xyz=D65, L_A=60, Y_b=20, surround=Average),
            XYZ(),
            color0=[0.246880325931, 0.484522681528, np.radians(299.40091582)],
            color1=[0.8006454338, 0.9505764278, 2.3723678665],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_XYZ_to_xyY(self):
        from vsl_ial.cs.ciexyy import CIExyY

        self._test(
            XYZ(),
            CIExyY(illuminant_xyz=D65),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.3597, 0.7163, 0.1193],
            ref0=[0.2441624864758532, 0.1473984459525917, 0.07493],
            ref1=[0.3009286371622187, 0.5992637831506735, 0.7163],
        )

    def test_xyY_to_XYZ(self):
        from vsl_ial.cs.ciexyy import CIExyY

        self._test(
            CIExyY(illuminant_xyz=D65),
            XYZ(),
            color0=[0.2441624864758532, 0.1473984459525917, 0.07493],
            color1=[0.3009286371622187, 0.5992637831506735, 0.7163],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.3597, 0.7163, 0.1193],
        )

    def test_XYZ_to_Opponent(self):
        from vsl_ial.cs.opponent import Opponent

        self._test(
            XYZ(),
            Opponent(),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[0.05548398, -0.05781628, 0.12142492],
            ref1=[0.6045871573, 0.03641176680, -0.332278089],
        )

    def test_Opponent_to_XYZ(self):
        from vsl_ial.cs.opponent import Opponent

        self._test(
            Opponent(),
            XYZ(),
            color0=[0.05548398, -0.05781628, 0.12142492],
            color1=[0.6045871573, 0.03641176680, -0.332278089],
            ref0=[0.12412, 0.07493, 0.3093],
            ref1=[0.36171007855, 0.7172837833, 0.11938578030],
        )

    def test_XYZ_to_PCS23UCS(self):
        from vsl_ial.cs.pcs23 import PCS23UCS

        self._test(
            XYZ(),
            PCS23UCS(illuminant_xyz=D65, F_LA_or_D=0.8),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[4.0039491820, 7.44766981859, -1.60553455705],
            ref1=[9.89742100222, -20.00986956453, -3.48618387085],
        )

        self._test(
            XYZ(),
            PCS23UCS(illuminant_xyz=None, F_LA_or_D=0.8),
            color0=[0.12412, 0.07493, 0.3093],
            color1=[0.36171007855, 0.7172837833, 0.11938578030],
            ref0=[3.75130767167, 7.2881813939, -1.915114599357],
            ref1=[9.60144074269, -21.14144774085, -4.807631138],
        )

    def test_cat02(self):
        from vsl_ial.cs.cam import CAT02

        cat02 = CAT02(
            illuminant_src=whitepoints_cie1964.D50,
            illuminant_dst=whitepoints_cie1964.D65,
            F_LA_or_D=0.8,
            exact=False,
        )
        self._test_f(
            cat02,
            [0.12412, 0.07493, 0.3093],
            [0.13432999186494882, 0.07917329191464952, 0.3865924699953725],
        )

        cat02 = CAT02(
            illuminant_src=whitepoints_cie1964.D50,
            illuminant_dst=whitepoints_cie1964.D65,
            F_LA_or_D=(Average.F, 110.0),
            exact=False,
        )
        self._test_f(
            cat02,
            [0.12412, 0.07493, 0.3093],
            [0.1362031329174131, 0.07995176043384412, 0.4007725940907399],
        )


class TestCaseCS1D(TestCaseCSBase, unittest.TestCase):
    def _test(
        self,
        src: CS,
        dst: CS,
        color0: npt.ArrayLike,
        color1: npt.ArrayLike,
        ref0: npt.ArrayLike,
        ref1: npt.ArrayLike,
    ):
        for color, ref in (color0, ref0), (color1, ref1):
            color = np.asarray(color)
            assert color.ndim == 1, color.ndim
            res_1d = convert(src, dst, color=color)
            np.testing.assert_almost_equal(res_1d, ref, decimal=4)

    def _test_f(
        self,
        f: Callable[[npt.ArrayLike], npt.ArrayLike],
        color: npt.ArrayLike,
        ref: npt.ArrayLike,
    ):
        res_1d = f(color)
        np.testing.assert_almost_equal(res_1d, ref, decimal=8)


class TestCaseCS2D(TestCaseCSBase, unittest.TestCase):
    def _test(
        self,
        src: CS,
        dst: CS,
        color0: npt.ArrayLike,
        color1: npt.ArrayLike,
        ref0: npt.ArrayLike,
        ref1: npt.ArrayLike,
    ):
        res_2d = convert(src, dst, color=[color0, color1])
        np.testing.assert_almost_equal(res_2d, [ref0, ref1], decimal=4)

    def _test_f(
        self,
        f: Callable[[npt.ArrayLike], npt.ArrayLike],
        color: npt.ArrayLike,
        ref: npt.ArrayLike,
    ):
        res_2d = f([color, color])
        np.testing.assert_almost_equal(res_2d, [ref, ref], decimal=8)


class TestCaseCS3D(TestCaseCSBase, unittest.TestCase):
    def _test(
        self,
        src: CS,
        dst: CS,
        color0: npt.ArrayLike,
        color1: npt.ArrayLike,
        ref0: npt.ArrayLike,
        ref1: npt.ArrayLike,
    ):
        res_3d = convert(
            src,
            dst,
            color=[[color0, color0, color1], [color1, color0, color1]],
        )
        np.testing.assert_almost_equal(
            res_3d, [[ref0, ref0, ref1], [ref1, ref0, ref1]], decimal=4
        )

    def _test_f(
        self,
        f: Callable[[npt.ArrayLike], npt.ArrayLike],
        color: npt.ArrayLike,
        ref: npt.ArrayLike,
    ):
        res_2d = f([[color, color, color], [color, color, color]])
        np.testing.assert_almost_equal(
            res_2d, [[ref, ref, ref], [ref, ref, ref]], decimal=8
        )
