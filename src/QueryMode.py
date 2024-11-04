from enum import Enum
class Mode(Enum):
    FAST = 1
    DENOISE = 2
    COMPRESS = 3
    PLOT_RUNTIME = 4

    @staticmethod
    def from_value(mode_input):
        try:
            return Mode(int(mode_input))
        except ValueError:
            raise ValueError(f"Invalid mode input: {mode_input}. Must be an integer between 1 and 4.")

