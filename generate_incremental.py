from utils.generators import ARGenerator, EnvelopeGenerator, ThicknessGenerator
from visualise import render_col

## generate single column of image
def generate_col(phi, noise_std, envelopes, height, carrier_std, amp=10.0):
    carrier = ARGenerator(phi, noise_std).step()
    envelope = EnvelopeGenerator(envelopes).step()

    y = amp * carrier * envelope
    thickness = ThicknessGenerator().step()
    print(
    f"carrier={carrier:.4f}, "
    f"envelope={envelope:.4f}, "
    f"amp_px={amp:.2f}, "
    f"y={y:.4f}"
    )
    col = render_col(height, y, thickness)

    return col