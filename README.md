# pandas-decimal

Extension array for pandas implementing fixed-precision decimal numbers.

Data is stored internally as integers and a dtype encoding the number of decimal
places. Operations are mostly implemented as integer arithmetic (i.e., fast!)
and always rounded to the required precision, meaning no floating point
quirks.

### Install

```bash
pip install git+//github.com/intake/pandas-decimal
```
