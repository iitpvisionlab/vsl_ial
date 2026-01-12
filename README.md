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
| `PCS23-UCS()` | Uniform Color Space with Advanced Hue Linearity |

### Available measures of the consistency between perceived and computed color differences
* CV
* PF/3
* STRESS
* Mean STRESS
* Group STRESS
* Weighted group STRESS

## Model evaluation

Run
```bash
python -m vsl_ial.eval
```

To get uniformity table

|           | COMBVD | BFD-P d65 | BFD-P m | BFD-P c | RIT-DuPont | Witt  | Leeds | Munsell-3.1.0 |
|-----------|--------|-----------|---------|---------|------------|-------|-------|---------------|
| CAM16-SCD | 0.319  | 0.298     | 0.358   | 0.285   | 0.213      | 0.29  | 0.251 | 0.293         |
| CAM16-UCS | 0.305  | 0.271     | 0.35    | 0.297   | 0.206      | 0.31  | 0.245 | 0.296         |
| CAM16-LCD | 0.337  | 0.293     | 0.378   | 0.376   | 0.251      | 0.397 | 0.303 | 0.327         |
| CAM02-SCD | 0.321  | 0.306     | 0.353   | 0.303   | 0.219      | 0.286 | 0.253 | 0.318         |
| CAM02-UCS | 0.306  | 0.28      | 0.343   | 0.321   | 0.213      | 0.305 | 0.246 | 0.326         |
| CAM02-LCD | 0.337  | 0.302     | 0.37    | 0.419   | 0.261      | 0.392 | 0.309 | 0.398         |
| PCS23-UCS | 0.311  | 0.289     | 0.325   | 0.379   | 0.3        | 0.381 | 0.332 | 0.337         |
| CIELAB    | 0.426  | 0.41      | 0.433   | 0.543   | 0.334      | 0.517 | 0.401 | 0.441         |
| ProLab    | 0.441  | 0.451     | 0.429   | 0.485   | 0.302      | 0.519 | 0.394 | 0.271         |
| Oklab     | 0.471  | 0.515     | 0.424   | 0.416   | 0.318      | 0.452 | 0.45  | 0.074         |
| JzAzBz    | 0.418  | 0.404     | 0.424   | 0.494   | 0.385      | 0.474 | 0.451 | 0.111         |

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
