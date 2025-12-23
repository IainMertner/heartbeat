from utils.generators import ARGenerator, EnvelopeGenerator, ThicknessGenerator
from visualise import render_col

## generate single column of image
def generate_col(ar_gen, env_gen, thick_gen, height, amp=10.0):
    # get next values from generators
    carrier = ar_gen.step()
    envelope = env_gen.step()

    y = amp * carrier * envelope
    thickness = thick_gen.step()
    col = render_col(height, y, thickness)

    return col