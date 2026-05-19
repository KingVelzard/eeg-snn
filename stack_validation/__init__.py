"""Stack validation for eeg-snn: eyes-open/closed pipeline end-to-end.

Submodules:
    montage     - 16-channel 10-20 layout constants.
    preprocess  - bandpass, notch, CAR, z-score, artifact rejection.
    encode      - delta spike encoder.
    model       - snnTorch LIF stack + leaky-integrator readout.
    loaders     - data source loaders (PhysioNet, OpenBCI).
    train       - training loop entry point.
"""
