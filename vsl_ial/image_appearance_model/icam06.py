from __future__ import annotations
from functools import cached_property
import numpy as np
from numpy.fft import fft2, ifft2
from .. import FArray


class iCAM06:
    ipt: FArray

    def __init__(self, ipt: FArray):
        self.ipt = ipt

    @cached_property
    def xyz(self) -> FArray:
        # invert chromatic adaptation
        return iCAM06_invcat(self.ipt)

    @cached_property
    def sRGB(self) -> FArray:
        return iCAM06_disp(self.xyz)


def idl_dist(m: int, n: int):
    x = np.arange(n)
    x = np.minimum(x, n - x) ** 2
    a = np.zeros((m, n))
    for i in range(m // 2 + 1):
        y = np.sqrt(x + i * i)
        a[i] = y
        if i:
            a[m - i] = y
    return a


def iCAM06_blur(img: FArray, d: int) -> FArray:
    sy, sx, sz = img.shape
    m = min(sy, sx)
    z: int = 1
    if m < 64:
        z = 1
    elif m < 256:
        z = 2
    elif m < 512:
        z = 4
    elif m < 1024:
        z = 8
    elif m < 2056:
        z = 16
    else:
        z = 32
    img = img[::z, ::z]
    yDim, xDim, _ = img.shape
    pad_t, pad_r, pad_b, pad_l = (
        yDim // 2,
        xDim - xDim // 2,
        yDim - yDim // 2,
        xDim // 2,
    )

    Y = np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="symmetric")
    assert Y.shape == (yDim * 2, xDim * 2, 3)
    # del img
    distMap = idl_dist(*Y.shape[0:2])

    # Gaussian filter
    Dim = max(xDim, yDim)
    kernel = np.exp(-1.0 * (distMap / (Dim / d)) ** 2)
    # del distMap

    filter = np.maximum(np.real(fft2(kernel)), 0)
    filter = filter / filter[0, 0]

    # since we are convolving, normalize the kernel to sum
    # to 1, and shift it to the center
    white = np.maximum(
        ifft2(fft2(Y, axes=(0, 1)) * filter[..., None], axes=(0, 1)).real, 0.0
    )
    white = white[yDim // 2 : yDim // 2 + yDim, xDim // 2 : xDim // 2 + xDim]

    # upsampling
    white = imresize_nearest(white, z)[0:sy, 0:sx]

    return white


def iCAM06_disp(XYZ_tm: FArray) -> FArray:
    XYZ_tm = XYZ_tm / np.max(XYZ_tm[:, :, 1])
    M = np.array(
        (
            (3.2407, -0.9693, 0.0556),
            (-1.5373, 1.8760, -0.2040),
            (-0.4986, 0.0416, 1.0571),
        ),
        dtype=np.float64,
    )
    RGB = changeColorSpace(XYZ_tm, M.T)
    # Clipping: simulate incomplete light adaptation and the glare in visual system
    # clip 1% dark pixels and light pixels individually
    min_rgb, max_rgb = np.percentile(RGB, (1, 99))
    min_rgb = max(min_rgb, 0.0)
    RGB = (RGB - min_rgb) / (max_rgb - min_rgb)
    RGB = RGB.clip(0, 1)
    # normalization
    sRGB = np.where(
        (RGB >= -0.0031308) & (RGB <= 0.0031308),
        RGB * 12.92,
        RGB ** (1.0 / 2.4) * 1.055 - 0.055,
    )
    outImage = np.uint8(sRGB * 255)
    return outImage


def iCAM06_HDR(
    xyz: FArray, p: float = 0.75, gamma_value: float = 1.0
) -> iCAM06:
    """
    Parameters
    ----------
    p : power value for tone compression
        (adjust contrast of the rendered image output).
        Can be set in a range [0.6,0.85],
        the larger value generates higher contrast.

    gamma_value : surround parameter (power value for I channel in IPT CS),
                  1.5 stands for dark surround, 1.25 -- dim, 1.0 -- average.
    """
    # separate Y into base-layer and detail-layer using bilateral filter
    base_img_x, detail_img_x = fastbilateralfilter(xyz[:, :, 0])
    base_img_y, detail_img_y = fastbilateralfilter(xyz[:, :, 1])
    base_img_z, detail_img_z = fastbilateralfilter(xyz[:, :, 2])
    base_img = np.dstack((base_img_x, base_img_y, base_img_z))
    detail_img = np.dstack((detail_img_x, detail_img_y, detail_img_z))

    # image chromatic adaptation
    # adaptation white for the base-layer padding the edge with flipped-border
    white = iCAM06_blur(xyz, 2)
    XYZ_adapt = iCAM06_CAT(base_img, white)

    # tone compression
    white = iCAM06_blur(xyz, 3)

    # local adaptation
    XYZ_tc = iCAM06_TC(XYZ_adapt, white, p)

    # combine the details
    XYZ_d = XYZ_tc * iCAM06_LocalContrast(detail_img, base_img)

    # transform into IPT space and color adjustment
    XYZ_p = iCAM06_IPT(XYZ_d, base_img, gamma_value)

    return iCAM06(XYZ_p)


def iCAM06_invcat(XYZ_img: FArray) -> FArray:
    # invert chromatic adaptation to display's white
    # written by Jiangtao (Willy) Kuang
    # Feb. 22, 2006

    # First things first...define the XYZ to RGB transform again using the CIECAM02 transform
    M = np.array(
        (
            (0.8562, 0.3372, -0.1934),
            (-0.8360, 1.8327, 0.0033),
            (0.0357, -0.0469, 1.0112),
        ),
        dtype=np.float64,
    ).T

    Mi = np.linalg.inv(M)

    xyz_d65 = np.array((95.05, 100.0, 108.88), dtype=np.float64)
    RGB_d65 = changeColorSpace(xyz_d65, M)

    # sRGB Output
    # For general PC user (Uncomment this part)
    whitepoint = np.array((95.05, 100.0, 108.88), dtype=np.float64)

    RGB_white = changeColorSpace(whitepoint, M)
    RGB_img = changeColorSpace(XYZ_img, M)

    # we want to use a complete adaptation transform, so
    # keep D set to 1.0, and don't try to calculate it
    D = 1

    adaptImage = (D * RGB_white / RGB_d65 + 1.0 - D) * RGB_img

    XYZ_adapt = changeColorSpace(adaptImage, Mi)
    return XYZ_adapt


def iCAM06_IPT(XYZ_img: FArray, base_img: FArray, gamma: float):
    # transform into IPT and post-processing
    # written by Jiangtao (Willy) Kuang
    # Feb. 22, 2006

    # transform into IPT space
    xyz2lms = np.array(
        (
            (0.4002, 0.7077, -0.0807),
            (-0.2280, 1.1500, 0.0612),
            (0.0, 0.0, 0.9184),
        ),
        dtype=np.float64,
    )
    iptMat = np.array(
        (
            (0.4000, 0.4000, 0.2000),
            (4.4550, -4.8510, 0.3960),
            (0.8056, 0.3572, -1.1628),
        ),
        dtype=np.float64,
    )

    # convert to LMS space
    lms_img = changeColorSpace(XYZ_img, xyz2lms)

    # apply the IPT exponent
    ipt_img = changeColorSpace(abs(lms_img) ** 0.43, iptMat)
    c = np.sqrt(ipt_img[:, :, 1] ** 2 + ipt_img[:, :, 2] ** 2)

    # colorfulness adjustment - Hunt effect
    La = 0.2 * base_img[:, :, 1]
    k = 1.0 / (5.0 * La + 1.0)
    FL = 0.2 * k**4.0 * (5.0 * La) + 0.1 * (1 - k**4) ** 2 * (
        5.0 * La
    ) ** (1 / 3)
    ipt_img[:, :, 1] = ipt_img[:, :, 1] * (
        (FL + 1.0) ** 0.15
        * ((1.29 * c**2 - 0.27 * c + 0.42) / (c**2 - 0.31 * c + 0.42))
    )
    ipt_img[:, :, 2] = ipt_img[:, :, 2] * (
        (FL + 1.0) ** 0.15
        * ((1.29 * c**2 - 0.27 * c + 0.42) / (c**2 - 0.31 * c + 0.42))
    )
    # to turn off the details enhancement, comment two lines above and uncomment two lines below
    # ipt_img(:,:,2) = ipt_img(:,:,2)
    # ipt_img(:,:,3) = ipt_img(:,:,3)

    # Bartleson surround adjustment
    max_i = np.max(ipt_img[:, :, 0])
    ipt_img[:, :, 0] = (ipt_img[:, :, 0] / max_i) ** gamma * max_i

    # inverse IPT
    lms_img = changeColorSpace(ipt_img, np.linalg.inv(iptMat))
    XYZ_p = changeColorSpace(
        abs(lms_img) ** (1 / 0.43), np.linalg.inv(xyz2lms)
    )
    return XYZ_p


def iCAM06_LocalContrast(detail: FArray, base_img: FArray) -> FArray:
    # steven's effect
    # written by Jiangtao (Willy) Kuang
    # Feb. 20, 2006

    La = 0.2 * base_img[:, :, 1]
    k = 1.0 / (5.0 * La + 1.0)
    FL = 0.2 * k**4 * (5.0 * La) + 0.1 * (1 - k**4) ** 2 * (5.0 * La) ** (
        1 / 3
    )
    # default parameter settings: a =0.25, b=0.8;
    detail_s = detail ** ((FL[..., None] + 0.8) ** 0.25)
    # to turn off the details enhancement, comment the line above and uncomment the line below
    # detail_s = detail;
    return detail_s


def iCAM06_TC(XYZ_adapt: FArray, white_img: FArray, p: float) -> FArray:
    # iCAM tone mapping based on Hunt color appearance model and CIECAM02
    # written by Jiangtao (Willy) Kuang
    # Feb. 20, 2006

    # transform the adapted XYZ to Hunt-Pointer-Estevez space
    M = np.array(
        (
            (0.38971, 0.68898, -0.07868),
            (-0.22981, 1.18340, 0.04641),
            (0.00000, 0.00000, 1.00000),
        ),
        dtype=np.float64,
    )
    Mi = np.linalg.inv(M)
    RGB_img = changeColorSpace(XYZ_adapt, M)

    # cone response
    La = 0.2 * white_img[:, :, 1]
    k = 1.0 / (5.0 * La + 1.0)
    FL = (
        0.2 * k**4.0 * (5.0 * La)
        + 0.1 * (1.0 - k**4.0) ** 2.0 * (5 * La) ** (1.0 / 3.0)
    )[..., None]

    # compression
    # default setting: p = .75
    sign_RGB = np.sign(RGB_img)
    Y = white_img[:, :, 1][..., None]
    RGB_c = (
        sign_RGB
        * (
            (400.0 * (FL * np.abs(RGB_img) / Y) ** p)
            / (27.13 + (FL * np.abs(RGB_img) / Y) ** p)
        )
        + 0.1
    )

    # make a netural As Rod response
    Las = 2.26 * La
    j = 0.00001 / (5 * Las / 2.26 + 0.00001)
    FLS = 3800 * j**2 * (5 * Las / 2.26) + 0.2 * (1 - j**2) ** 4 * (
        5 * Las / 2.26
    ) ** (1 / 6)
    Sw = np.max(5 * La, axis=0)
    S = np.abs(np.abs(XYZ_adapt[:, :, 1]))
    Bs = 0.5 / (1.0 + 3.0 * ((5 * Las / 2.26) * (S / Sw)) ** 3) + 0.5 / (
        1 + 5 * (5 * Las / 2.26)
    )
    # Noise term in Rod response is 1/3 of that in Cone response because Rods are more sensitive
    As = (
        3.05
        * Bs
        * (((400.0 * (FLS * (S / Sw)) ** p) / (27.13 + (FLS * (S / Sw)) ** p)))
        + 0.03
    )

    # combine Cone and Rod response
    RGB_c = RGB_c + As[..., None]

    # convert RGB_c back to XYZ space
    XYZ_tc = changeColorSpace(RGB_c, Mi)
    return XYZ_tc


def fastbilateralfilter(img: FArray) -> tuple[FArray, FArray]:
    if min(img.shape[0:2]) < 1024:
        z = 2
    else:
        z = 4
    img[img < 0.0001] = 0.0001
    logimg = np.log10(img)
    base_layer = PiecewiseBilateralFilter(logimg, z)

    # remove error points if any
    base_layer = np.minimum(base_layer, logimg.max())
    detail_layer = logimg - base_layer
    detail_layer[detail_layer > 12] = 0
    base_layer = 10**base_layer
    detail_layer = 10**detail_layer

    return base_layer, detail_layer


def imresize_nearest(img: FArray, d: int):
    return img.repeat(d, axis=1).repeat(d, axis=0)


def PiecewiseBilateralFilter(imageIn: FArray, z: int):
    yDim, xDim = imageIn.shape[0:2]
    sigma_s = 2 * xDim / z / 100
    sigma_r = 0.35
    maxI, minI = imageIn.max(), imageIn.min()
    nSeg = (maxI - minI) / sigma_r
    inSeg = round(nSeg)
    if inSeg == 0.0:  # vsl_ial fix to avoid division by zero
        return imageIn  # vsl_ial fix

    # Create Gaussian Kernel
    distMap = idl_dist(yDim, xDim)
    kernel = np.exp(-1 * (distMap / sigma_s) ** 2)
    kernel = kernel / kernel[0, 0]
    fs = np.maximum(np.real(fft2(kernel)), 0)
    fs = fs / fs[0, 0]

    # downsampling
    Ip = imageIn[::z, ::z]
    fsp = fs[::z, ::z]

    # Set the output to zero
    imageOut = np.zeros_like(imageIn)

    for j in range(inSeg + 1):
        value_i = minI + j * (maxI - minI) / inSeg
        # edge-stopping function
        jGp = np.exp((-1 / 2) * ((Ip - value_i) / sigma_r) ** 2)
        # normalization factor
        jKp = np.maximum(np.real(ifft2(fft2(jGp[:, :]) * fsp)), 0.0000000001)
        # Compute H for each pixel
        jHp = jGp * Ip
        sjHp = np.real(ifft2(fft2(jHp[:, :]) * fsp))
        # normalize
        jJp = sjHp / jKp

        #  upsampling
        # jJ = imresize(jJp, z, 'nearest')
        # jJ = jJ[0:yDim, 0:xDim]
        jJ = imresize_nearest(jJp, z)[0:yDim, 0:xDim]

        # interpolation
        intW = np.maximum(
            np.ones_like(imageIn)
            - np.abs(imageIn - value_i) * (inSeg) / (maxI - minI),
            0,
        )
        #
        imageOut[:] = imageOut + jJ * intW
    return imageOut


def changeColorSpace(img: FArray, m: FArray) -> FArray:
    return np.tensordot(img, m.T, axes=1)


def iCAM06_CAT(XYZimg: FArray, white: FArray) -> FArray:
    assert XYZimg.ndim == 3
    assert XYZimg.shape == white.shape, (XYZimg.shape, white.shape)
    # iCAM color chromatic adaptation
    # use CIECAM02 CAT here
    # written by Willy Kuang
    # Feb. 20, 2006

    # First things first...define the XYZ to RGB transform
    M = np.array(
        (
            (0.7328, 0.4296, -0.1624),
            (-0.7036, 1.6974, 0.0061),
            (0.0030, 0.0136, 0.9834),
        ),
        dtype=np.float64,
    )
    Mi = np.linalg.inv(M)
    RGB_img = changeColorSpace(XYZimg, M)

    RGB_white = changeColorSpace(white, M)
    xyz_d65 = np.array((95.05, 100.0, 108.88), dtype=np.float64)
    RGB_d65 = changeColorSpace(xyz_d65, M)

    La = 0.2 * white[:, :, 1]
    # % CIECAM02 CAT
    # % suppose it is in an average surround
    F = 1.0
    # default setting for 30% incomplete chromatic adaptation: a = 0.3
    D = 0.3 * F * (1.0 - (1.0 / 3.6) * np.exp((La - 42.0) / -92.0))

    RGB_white = RGB_white + 0.0000001
    adaptImage = (
        np.tensordot(D, RGB_d65, axes=0) / RGB_white + 1.0 - D[..., None]
    ) * RGB_img
    XYZ_adapt = changeColorSpace(adaptImage, Mi)
    return XYZ_adapt


def rgb_to_xyz_like_in_article(rgb: FArray, max_L: float = 0.0) -> FArray:
    """
    max_L: maximum luminance of the input image, in cd/m2 or nits.
    """
    M = np.array(
        (
            (0.412424, 0.212656, 0.0193324),
            (0.357579, 0.715158, 0.119193),
            (0.180464, 0.0721856, 0.950444),
        ),
        dtype=np.float64,
    )

    XYZimg = np.tensordot(rgb, M, axes=1)
    if max_L:
        XYZimg = XYZimg / XYZimg[:, :, 1].max() * max_L
    XYZimg[XYZimg < 0.00000001] = 0.00000001
    return XYZimg


def main():
    # load
    from OpenImageIO import ImageBuf, ImageOutput, geterror, ImageSpec, UINT8

    img = ImageBuf("PeckLake.hdr")
    rgb: FArray = img.get_pixels().copy()

    # process
    xyz = rgb_to_xyz_like_in_article(rgb, 20000.0)
    outImage = iCAM06_HDR(xyz, 0.7, 1).sRGB
    assert outImage.ndim == 3, outImage.ndim

    # save
    output = ImageOutput.create("PeckLake.png")
    specification = ImageSpec(outImage.shape[1], outImage.shape[0], 3, UINT8)
    output.open("PeckLake.png", specification)
    if not output:
        raise ValueError(f"PeckLake.png {geterror()}")
    if not output.write_image(outImage.copy()):
        raise ValueError(f"no way: {output.geterror()}")
    output.close()


if __name__ == "__main__":
    main()
