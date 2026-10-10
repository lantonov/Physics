import numpy as np
import matplotlib.pyplot as plt

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries

from pycbc.types import TimeSeries as PycbcTimeSeries
from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter
from pycbc.psd import welch, interpolate, inverse_spectrum_truncation
from lal import LIGOTimeGPS   # add this import at top if not already
from pycbc.psd import aLIGOZeroDetHighPower


def theta_phase_correction(frequencies, f0=20.0, theta_amp=1.0):
    hbar = 1.054571817e-34
    G   = 6.67430e-11
    c   = 299792458.0
    coeff = theta_amp * hbar * G / c**5
    safe_freq = np.maximum(frequencies, f0 * 1.01)
    return coeff * frequencies**2 * np.log(safe_freq / f0)


def generate_theta_waveform(m1, m2, delta_f, f_lower=20.0, theta_amp=1.0):
    hp, _ = get_fd_waveform(approximant="IMRPhenomPv2",
                            mass1=m1, mass2=m2,
                            delta_f=delta_f,
                            f_lower=f_lower)
    freqs = hp.sample_frequencies.data
    phase_corr = theta_phase_correction(freqs, theta_amp=theta_amp)
    hp.data *= np.exp(1j * phase_corr)
    return hp


def run_pipeline(event="GW150914", detector="H1", theta_amp=1.0):
    print(f"Loading ~30 s of strain data for {event} — {detector}")

    gps = event_gps(event)
    print(f"Event GPS time: {gps:.3f} s")

    start = int(gps) - 15
    end   = int(gps) + 15

    ts = TimeSeries.fetch_open_data(detector, start, end,
                                    sample_rate=4096,
                                    cache=True,
                                    verbose=True)
    print(f"Fetched {len(ts)} samples ({len(ts)*ts.dt.value:.1f} s)")

    if len(ts) == 0:
        raise ValueError("Fetch returned empty data.")

    # Bandpass
    ts = ts.bandpass(20, 1024)

    # After ts = ts.bandpass(...)
    crop_duration = 7.0
    ts_cropped = ts.crop(ts.span[0] + crop_duration, ts.span[1] - crop_duration)

    if len(ts_cropped) == 0:
        print("Warning: crop resulted in empty series — using uncropped bandpass data")
        ts_cropped = ts

    print(f"After crop: {len(ts_cropped)} samples, duration {ts_cropped.duration.value:.1f} s")

    # Epoch conversion
    from lal import LIGOTimeGPS
    epoch_gps = LIGOTimeGPS(ts_cropped.epoch.value)

    data = PycbcTimeSeries(
        ts_cropped.value,
        delta_t = ts_cropped.dt.value,
        epoch    = epoch_gps
    )

    # PSD estimation ────────────────────────────────────────────────
    fs = 1.0 / data.delta_t
    seg_len    = int(4 * fs)
    seg_stride = int(2 * fs)

    psd_welch_interp = welch(data, seg_len=seg_len, seg_stride=seg_stride)
    print("PSD median (20–300 Hz):",
    np.median(psd_welch_interp[np.logical_and(psd_welch_interp.sample_frequencies >= 20,
                                 psd_welch_interp.sample_frequencies <= 300)]))
    psd_welch_interp = interpolate(psd_welch_interp, data.delta_f)

    psd_truncated = inverse_spectrum_truncation(psd_welch_interp, int(16 * fs),   # try 16 s, or 24 s = int(24*fs)
                                  low_frequency_cutoff=20.0,
                                  trunc_method='hann')   # optional: smoother window
    # After truncation
    low_freq_mask = psd_truncated.sample_frequencies < 40.0
    psd_truncated.data[low_freq_mask] = np.maximum(psd_truncated.data[low_freq_mask], 1e-45)

    psd_analytic = aLIGOZeroDetHighPower(len(psd_truncated), psd_truncated.delta_f, 20.0)
    psd = 0.6 * psd_truncated + 0.4 * psd_analytic  # blend; tune weight

    # Template
    template = generate_theta_waveform(36, 29,
                                       delta_f=data.delta_f,
                                       f_lower=20.0,
                                       theta_amp=theta_amp)
    #print("Template norm:", template.norm())           # should be ~20–25
    template.resize(len(data) // 2 + 1)

    # Matched filter
    snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=20.0)

    # Peak
    peak_idx = np.argmax(np.abs(snr))
    peak_snr = np.abs(snr[peak_idx])
    peak_time = snr.sample_times[peak_idx]

    print(f"Peak SNR: {peak_snr:.2f}")
    print(f"Peak at GPS {peak_time:.3f} s (offset from merger: {peak_time - gps:.3f} s)")

    return snr, gps, psd_welch_interp, psd_truncated, template, psd, peak_idx


if __name__ == "__main__":
    snr_phys, merger_gps, psd_before, psd_after, template, psd, peak_idx = run_pipeline(theta_amp=1.0)

    times = snr_phys.sample_times - merger_gps

    plt.figure(figsize=(10, 6))
    plt.plot(times, np.abs(snr_phys), label="|SNR| (physical Θ ≈ standard)")
    plt.xlim(-0.3, 0.1)
    plt.axvline(0, color='r', ls='--', label='Merger time')
    plt.xlabel("Time from merger GPS (s)")
    plt.ylabel("|SNR|")
    plt.title("Matched-filter SNR — GW150914 H1")
    plt.grid(True)
    plt.legend()
    plt.show()