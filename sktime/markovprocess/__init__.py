import pint

ureg = pint.UnitRegistry()
ureg.define('step = []')  # dimensionless unit for unspecified lag time unit.
Q_ = ureg.Quantity

del pint
