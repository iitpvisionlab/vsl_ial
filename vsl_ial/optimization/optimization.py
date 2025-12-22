from __future__ import annotations
from typing import Callable, TypeAlias, Any, Literal
from pathlib import Path
from collections.abc import Sequence

import json5
from .config import Config
from vsl_ial import FArray
from vsl_ial.cs.pcs23 import PCS23UCS, CS
from vsl_ial.datasets.sensitivities import load as load_sensitivity
from vsl_ial.stress import Ord
from vsl_ial.cs.xyz import XYZ
from vsl_ial.cs.srgb import sRGB
from vsl_ial.cs.linrgb import linRGB
from vsl_ial.cs.ciexyy import CIExyY
from vsl_ial.cs import convert
from scipy.optimize import minimize
from .dataset import WeightedDataset
import numpy as np
import viser


Float = np.floating[Any]
F32Array = np.ndarray[Any, np.dtype(np.float32)]

LossFunction: TypeAlias = Callable[
    [Sequence[FArray], Sequence[FArray], Ord], Float
]

server = viser.ViserServer()


def _point_inside(x: float, y: float, poly: list[tuple[float, float]]):
    """
    ToDo: this is a very slow function.
          Can be optimized 100x after this module is working
    """
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def plot_munsell(dataset: WeightedDataset):
    from matplotlib import pyplot as plt

    ax = plt.gca()
    ax.set_aspect("equal")
    ax.grid()
    for ds in dataset.datasets:
        ax.scatter(*zip(*ds.xy))
    plt.show()


class MonotonicityLoss:
    def __init__(self) -> None:
        full_xy_grid = self._create_grid(
            np.linspace(0, 1, 30, dtype=np.float32),
            np.linspace(0, 1, 30, dtype=np.float32),
        )
        self.Y = Y = np.linspace(0.02, 1.0, 49)
        self._sensitivity_xyz = sensitivity_xyz = np.asarray(
            load_sensitivity("cie-1931-2")["xyz"], dtype=np.float64
        ).T
        spectral_locus_xy = (
            CIExyY().from_XYZ(XYZ(), sensitivity_xyz)[:, :2].tolist()
        )
        xy_points = np.asarray(
            [
                pt
                for pt in full_xy_grid
                if _point_inside(*pt.tolist(), spectral_locus_xy)
            ],
            dtype=np.float32,
        )
        self.xy_points = xy_points
        xy_points_repeated = xy_points[None].repeat(Y.size, 0)
        n_points = len(xy_points)
        xyY = np.dstack(
            (xy_points_repeated, Y.repeat(n_points).reshape(-1, n_points, 1))
        )
        self.XYZ_diff = CIExyY().to_XYZ(XYZ(), xyY)
        self.xyY_diff = xyY

    def __call__(self, cs: CS) -> float:
        """
        Formula 10 of PCS23-UCS
        """
        self.visualize_PCS23 = res = cs.from_XYZ(XYZ(), self.XYZ_diff)
        minimum_diff = 0.02
        L_plus = res[..., 2]
        θ = float(
            np.mean(
                np.maximum(
                    0,
                    -np.diff(L_plus, axis=0) / np.diff(self.Y)[..., None]
                    + minimum_diff,
                )
            )
        )
        return θ

    @staticmethod
    def _create_grid(x: F32Array, y: F32Array) -> F32Array:
        return np.dstack(np.meshgrid(x, y, indexing="ij")).reshape(-1, 2)


def create_sRGB_grid(samples: int = 100):
    r = np.linspace(0, 1, samples, dtype=np.float32)
    g = np.linspace(0, 1, samples, dtype=np.float32)
    b = np.linspace(0, 1, samples, dtype=np.float32)

    rgb_grid = np.array(np.meshgrid(r, g, b))
    mask = np.ones((samples, samples, samples), dtype=np.bool)
    mask[1:-1, 1:-1, 1:-1] = False

    srgb = rgb_grid.T.reshape(-1, 3)
    srgb = np.ascontiguousarray(srgb[mask.reshape(-1)])
    from vsl_ial.cs.srgb import sRGB

    xyz = sRGB().to_XYZ(sRGB(), srgb)

    return srgb, xyz


def train(
    model_cls: type[PCS23UCS],
    loaded_datasets: list[WeightedDataset],
    loss_function: LossFunction,
) -> None:
    monotonicity_loss = MonotonicityLoss()

    RGB_colors = (
        linRGB()
        .from_XYZ(None, monotonicity_loss._sensitivity_xyz)
        .reshape(-1, 3)
    )

    server.scene.add_point_cloud(
        name="/sensitivity",
        points=monotonicity_loss._sensitivity_xyz,
        colors=RGB_colors,
        point_size=0.03,
    )

    DBG = monotonicity_loss.XYZ_diff
    DBG = DBG[len(DBG) // 2]
    x, y, _ = (
        DBG.reshape(-1, 3) / np.sum(DBG.reshape(-1, 3), axis=-1)[:, None]
    ).T
    xyY = np.column_stack((x, y, np.full_like(x, fill_value=0.4)))

    RGB_colors_2 = convert(
        src=CIExyY(),
        dst=linRGB(),
        color=DBG.reshape(-1, 3),
    )

    server.scene.add_point_cloud(
        name="/XYZ_diff",
        points=xyY,
        colors=RGB_colors_2,
        point_size=0.01,
    )

    def evaluate(x: list[float]) -> float:

        opt_model = model_cls(
            F_LA_or_D=0.0, illuminant_xyz=None, V=x[:39], H=x[39:]
        )

        stress = 0.0
        for loaded_dataset in loaded_datasets:
            ref: list[FArray] = []
            exp: list[FArray] = []
            # print("=" * 40)
            for dataset in loaded_dataset.datasets:
                assert dataset.F is not None, dataset
                model = model_cls(
                    F_LA_or_D=(dataset.F, dataset.L_A),
                    illuminant_xyz=dataset.illuminant,
                    V=x[:39],
                    H=x[39:],
                )

                srgb, xyz = create_sRGB_grid(50)
                pc_coordinates = model.from_XYZ(XYZ(), xyz)

                from matplotlib import pyplot as plt

                if False:
                    from vsl_ial.cs.cielab import CIELAB
                    from vsl_ial.cs.cam import CAM16UCS, Surround

                    opt_model = model = CAM16UCS(
                        illuminant_xyz=np.array(dataset.illuminant),
                        L_A=dataset.L_A,
                        Y_b=dataset.Y_b,
                        surround=Surround(
                            F=dataset.F,
                            c=dataset.c,
                            N_c=dataset.Nc,
                        ),
                    )
                    assert dataset.Nc is not None, dataset.name

                    cam16ucs = model.from_XYZ(XYZ(), xyz)
                # x_plus, y_plus = zip(
                #     *(pc_coordinates[:, 1:] / pc_coordinates[:, :1])
                # )
                # plt.gca().set_ylim(5, -6)
                # plt.gca().grid()
                # plt.gca().set_aspect("equal")
                # plt.scatter(y_plus, x_plus)
                # plt.show()

                if False:
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection="3d")

                    ax.scatter(
                        pc_coordinates[:, 0],
                        pc_coordinates[:, 1],
                        pc_coordinates[:, 2],
                        c=srgb,
                        marker="o",
                        s=4,
                    )

                    # Set labels and title
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")
                    ax.set_title("sRGB Gamut in CIE XYZ Color Space")
                    ax.set_box_aspect([1, 1, 1])
                    plt.show()
                    return

                model_coordinates = model.from_XYZ(XYZ(), dataset.xyz)
                a_colors = np.empty((len(dataset.pairs), 3), dtype=np.float64)
                b_colors = np.empty_like(a_colors)
                for idx, (a_idx, b_idx) in enumerate(dataset.pairs):
                    a_colors[idx] = model_coordinates[a_idx]
                    b_colors[idx] = model_coordinates[b_idx]
                exp_distance = np.linalg.norm(
                    a_colors - b_colors, axis=1, ord=2
                )
                breakpoint()
                exp.append(exp_distance)
                ref.append(dataset.dv)
                # print(f"{dataset.name} -> {len(dataset.pairs)}")
            stress += loss_function(ref, exp) * loaded_dataset.weight
        # plot_munsell(loaded_dataset)
        monotonicity = monotonicity_loss(opt_model)
        loss = stress + 0.1 * monotonicity
        print(f"{stress=}, {monotonicity=}, {loss=}")
        # breakpoint()

        #
        # visualize
        #

        vis_pcs23 = opt_model.from_XYZ(
            XYZ(), monotonicity_loss._sensitivity_xyz
        )
        server.scene.add_point_cloud(
            name="/pcs23",
            points=vis_pcs23,
            colors=RGB_colors,
            point_size=0.03,
        )
        return loss

    # x0 = [0.2] * (39 + 8)
    # x0 = np.random.rand(39 + 8)
    # x0 = np.asarray(
    #     [
    #         5.62377545903092,
    #         171.36368020069722,
    #         1139.5284167730447,
    #         -0.0020158707793063793,
    #         -518.7569550219694,
    #         -137924.3491321199,
    #         -778.9657975296375,
    #         -27033.789978479195,
    #         -9803.553621070627,
    #         231.03532437905437,
    #         644.7093156808708,
    #         0.44725829317373944,
    #         1552.67428101527,
    #         12385.40992041996,
    #         -1.3479048145608292,
    #         175294.8272608068,
    #         217.6718130425752,
    #         -100733.5963017536,
    #         -551.1602008733025,
    #         1.8733545835396264,
    #         3.2401865056285963,
    #         -5.037047974757915,
    #         904.8509895568436,
    #         0.8076667483450888,
    #         17.92102636270046,
    #         -172.92820638970534,
    #         -30.23187778315493,
    #         -9447.097080971176,
    #         2405.2096148027103,
    #         38.666799865781016,
    #         -29.211639266195263,
    #         -1034.5453785483978,
    #         99.82711165184196,
    #         3.3476225751147854,
    #         -427.2206444638241,
    #         23555.99281961212,
    #         -67.88828249203013,
    #         940.1148914054388,
    #         -92.37431872163927,
    #         373.63658287545445,
    #         -93.78291385820899,
    #         -82.47410662677822,
    #         -85.44966655764117,
    #         382.8830618784323,
    #         -114.0144951964989,
    #         -42.61604476329086,
    #         171.52341864142795,
    #     ]
    # )
    # np.random.shuffle(x0)
    # x0 = -x0 + (x0 + 1) * 0.2
    x0 = np.array(PCS23UCS.DEFAULT_V + PCS23UCS.DEFAULT_H)
    for i in range(5):
        res = minimize(
            fun=evaluate,
            method="Nelder-Mead",
            x0=x0,
            tol=1e-2,
            options={"maxiter": 150, "maxfev": 150},
        )
        print(f"step {i+1}: ", res.x.tolist())
        x0 = res.x
    print("minimization result", res)


# def get_xyz():
#     with open(
#         "/home/senyai/projects/vsl_ial/vsl_ial/datasets/ciexyz31_1.csv"
#     ) as f:
#         lines: list[tuple[int, float, float, float]] = []
#         for line in f:
#             wl, x, y, z = line.split(",")
#             lines.append((int(wl), float(x), float(y), float(z)))
#     return lines


def main():
    #     wl_xyz = get_xyz()
    #     xyz = np.asarray([row[1:] for row in wl_xyz])
    #     xs, ys, zs = xyz.T

    #     x = [ 5.65822087e+00,  1.75511708e+02,  1.16367546e+03, -2.00084521e-03,
    # -5.05773364e+02, -1.36521727e+05, -5.89062148e+02, -2.64563249e+04,
    # -9.57581167e+03,  2.35986631e+02,  6.55043332e+02,  4.46391903e-01,
    #  1.58078875e+03,  1.24616075e+04, -1.34185422e+00,  1.78546715e+05,
    #  2.19355970e+02, -9.81213831e+04, -5.46287245e+02,  1.88035463e+00,
    #  3.18578603e+00, -5.09896100e+00,  8.85655496e+02,  7.94394362e-01,
    #  1.76772051e+01, -1.73650355e+02, -3.29927222e+01, -8.72088578e+03,
    #  2.35704637e+03,  3.77036132e+01, -2.88774290e+01, -1.03504007e+03,
    #  9.84241706e+01,  3.32752406e+00, -4.28197266e+02,  2.31725935e+04,
    # -6.81158416e+01,  9.27573345e+02, -9.31335222e+01,  3.79644431e+02,
    # -9.22254559e+01, -8.04988702e+01, -8.18506374e+01,  3.74275903e+02,
    # -1.15349017e+02, -4.35768649e+01,  1.68770308e+02]
    #     model = PCS23UCS()

    #     ucs = model.from_XYZ(None, xyz)
    #     xs, ys, zs = ucs.T

    #     import matplotlib.pyplot as plt

    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(xs, ys, zs)
    #     plt.show()

    #     return
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--update-schema", action="store_true")
    args = parser.parse_args()
    if args.update_schema:
        schema_path = Path(__file__).with_name("schema.json")
        schema_path.write_text(
            json5.dumps(
                Config.model_json_schema(), ensure_ascii=False, quote_keys=True
            )
        )
        return
    config = Config(**json5.loads(args.config.read_text()))
    loaded_datasets = [dataset.load() for dataset in config.datasets]
    loss = config.loss.load()
    model = PCS23UCS
    for loaded_dataset in loaded_datasets:
        print(f"* {loaded_dataset.name}")
        if len(loaded_dataset.datasets) != 1:
            for subset in loaded_dataset.datasets:
                print(f" - {subset.name}")

    print(f"model = {model}")
    print(f"loss = {loss}")
    train(model, loaded_datasets, loss)
