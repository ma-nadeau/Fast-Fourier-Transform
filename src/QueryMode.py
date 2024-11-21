from enum import Enum
class Mode(Enum):
    FAST = 1
    DENOISE = 2
    COMPRESS = 3
    PLOT_RUNTIME = 4

    @classmethod
    def from_value(cls, value: int) -> "Mode":
        if value not in [typ.value for typ in Mode]:
            raise ValueError(f"Invalid mode input: {value}. Must be an integer between 1 and 4.")
        return cls(value)
