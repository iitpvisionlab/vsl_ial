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

|           | COMBVD | BFD-P d65 | BFD-P m | BFD-P c | RIT-DuPont | Witt  | Leeds | Munsell-3.1.0* |
|-----------|--------|-----------|---------|---------|------------|-------|-------|----------------|
| CAM16-SCD | 0.295  | 0.254     | 0.346   | 0.283   | 0.237      | 0.306 | 0.219 | 0.0871         |
| CAM16-UCS | 0.305  | 0.271     | 0.35    | 0.297   | 0.206      | 0.31  | 0.245 | 0.0883         |
| CAM16-LCD | 0.339  | 0.311     | 0.372   | 0.366   | 0.214      | 0.372 | 0.292 | 0.107          |
| CAM02-SCD | 0.296  | 0.266     | 0.338   | 0.303   | 0.244      | 0.303 | 0.221 | 0.111          |
| CAM02-UCS | 0.306  | 0.28      | 0.343   | 0.321   | 0.213      | 0.305 | 0.246 | 0.116          |
| CAM02-LCD | 0.339  | 0.318     | 0.366   | 0.404   | 0.223      | 0.366 | 0.296 | 0.169          |
| PCS23-UCS | 0.311  | 0.289     | 0.325   | 0.379   | 0.3        | 0.381 | 0.332 | 0.0741         |
| CIELAB    | 0.426  | 0.41      | 0.433   | 0.543   | 0.334      | 0.517 | 0.401 | 0.281          |
| ProLab    | 0.441  | 0.451     | 0.429   | 0.485   | 0.302      | 0.519 | 0.394 | 0.177          |
| Oklab     | 0.471  | 0.515     | 0.424   | 0.416   | 0.318      | 0.452 | 0.45  | 0.074          |
| JzAzBz    | 0.418  | 0.404     | 0.424   | 0.494   | 0.385      | 0.474 | 0.451 | 0.127          |
| ICaCb     | 0.391  | 0.396     | 0.38    | 0.424   | 0.248      | 0.474 | 0.373 | 0.14           |
| ICtCp     | 0.463  | 0.481     | 0.441   | 0.636   | 0.423      | 0.559 | 0.397 | 0.228          |
\* subset of Munsell dataset

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
