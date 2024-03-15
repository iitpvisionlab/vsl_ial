# VSL Image Appearance Library

Python module for working with color spaces and stuff

## Usage

### Color space conversion:

* Convert from `sRGB` to `libRGB`

```python
from vsl_ial.cs import convert, sRGB, linRGB

convert(sRGB(), linRGB(), color=[0.12412, 0.07493, 0.3093])
```

### Available color spaces

| Function name | Color space description |
| :---          | :--- |
| `XYZ()`       | CIE XYZ |
| `CIExyY()`    | CIE xyY |
| `CIELUV()`    | CIELUV |
| `CIELAB()`    | CIELAB |
| `ProLab()`    | proLab |
| `Oklab()`     | Oklab |
| `ICaCb()`     | ICaCb |
| `ICtCp()`     | ICtCp |
| `JzAzBz()`    | Jzazbz |
| `CAM02LCD()`  | CAM02-LCD |
| `CAM02SCD()`  | CAM02-SCD |
| `CAM02UCS()`  | CAM02-UCS |
| `CAM16LCD()`  | CAM16-LCD |
| `CAM16SCD()`  | CAM16-SCD |
| `CAM16UCS()`  | CAM16-UCS |
| `sRGB()`      | sRGB |
| `linRGB()`    | linear sRGB |
| `LMS()`       | LMS |
| `Opponent()`  | Opponent color space by Zhang and Wandell |

### Available measures of the consistency between perceived and computed color differences
* CV
* PF/3
* STRESS
* Mean STRESS
* Group STRESS
* Weighted group STRESS

## For module developers

* How to make whl file:

```bash
python -m pip install --upgrade build
python -m build
```


* How to run unit tests:

```bash
python -m unittest
```

* How to run coverage:

```bash
python -m coverage run -m unittest
python -m coverage html
```
