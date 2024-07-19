from typing import Union

import numpy as np
from scipy import signal as sg


class PanTompkinsPreprocess():

    def low_pass_filter(self, signal: np.ndarray) -> np.ndarray:
        """ Original low pass filter.
            The low pass filter has the recursive equation:
            y(nT) = 2y(nT - T) - y(nT - 2T) + x(nT) - 2x(nT - 6T) + x(nT - 12T)

        Args:
            signal (np.ndarray): input signal

        Returns:
            np.ndarray: filtered signal
        """
        filtered_signal = signal.copy()
        for index in range(len(signal)):
            filtered_signal[index] = signal[index]

            if (index >= 1):
                filtered_signal[index] += 2 * filtered_signal[index - 1]

            if (index >= 2):
                filtered_signal[index] -= filtered_signal[index - 2]

            if (index >= 6):
                filtered_signal[index] -= 2 * signal[index - 6]

            if (index >= 12):
                filtered_signal[index] += signal[index - 12]
        return filtered_signal

    def high_pass_filter(self, signal: np.ndarray) -> np.ndarray:
        """ Original high pass filter
            The high pass filter has the recursive equation:
            y(nT) = 32x(nT - 16T) - y(nT - T) - x(nT) + x(nT - 32T)

        Args:
            signal (np.ndarray): input signal

        Returns:
            np.ndarray: filtered signal
        """

        filtered_signal = signal.copy()
        for index in range(len(signal)):
            filtered_signal[index] = -1 * signal[index]

            if (index >= 1):
                filtered_signal[index] -= filtered_signal[index - 1]

            if (index >= 16):
                filtered_signal[index] += 32 * signal[index - 16]

            if (index >= 32):
                filtered_signal[index] += signal[index - 32]

        return filtered_signal

    def low_pass_filter_updated(self, signal: np.ndarray) -> np.ndarray:
        """ Updated low pass filter
            The low pass filter has the recursive equation:
             y(nT)=2y(nT-T)-y(nT-2T)+x(nT)-2x(nT-5T)+x(nT-10T)

        Args:
            signal (np.ndarray): input signal

        Returns:
            np.ndarray: filtered signal
        """
        filtered_signal = np.zeros_like(signal)
        for n in range(len(signal)):
            if n >= 10:
                filtered_signal[n] = (2 * filtered_signal[n - 1] - filtered_signal[n - 2] + signal[n] -
                                      2 * signal[n - 5] + signal[n - 10])
            elif n >= 5:
                filtered_signal[n] = (2 * filtered_signal[n - 1] - filtered_signal[n - 2] + signal[n] -
                                      2 * signal[n - 5])
            elif n >= 2:
                filtered_signal[n] = (2 * filtered_signal[n - 1] - filtered_signal[n - 2] + signal[n])
            elif n >= 1:
                filtered_signal[n] = (2 * filtered_signal[n - 1] + signal[n])
            else:
                filtered_signal[n] = signal[n]

        return filtered_signal

    def high_pass_filter_updated(self, signal: np.ndarray) -> np.ndarray:
        """ Updated high pass filter
            The high pass filter has the recursive equation:
            y(n) = y(n-1) - (1/32)x(n) + x(n-16) - x(n-17) + (1/32)x(n-32)

        Args:
            signal (np.ndarray): input signal

        Returns:
            np.ndarray: filtered signal
        """

        filtered_signal = np.zeros_like(signal)
        for n in range(len(signal)):
            if n >= 32:
                filtered_signal[n] = (filtered_signal[n - 1] - (1 / 32) * signal[n] + signal[n - 16] - signal[n - 17] +
                                      (1 / 32) * signal[n - 32])
            elif n >= 17:
                filtered_signal[n] = (filtered_signal[n - 1] - (1 / 32) * signal[n] + signal[n - 16] - signal[n - 17])
            elif n >= 16:
                filtered_signal[n] = (filtered_signal[n - 1] - (1 / 32) * signal[n] + signal[n - 16])
            elif n >= 1:
                filtered_signal[n] = (filtered_signal[n - 1] - (1 / 32) * signal[n])
            else:
                filtered_signal[n] = -(1 / 32) * signal[n]

        return filtered_signal

    def band_pass_filter(self, signal: np.ndarray, updated: bool = True) -> np.ndarray:
        """ Band Pass Filter.
            It is used to attenuate the noise in the input signal.
            It is the combination of a low pass filter with a high pass filter.

        Args:
            signal (np.ndarray): input signal
            updated (bool, optional): use the updated version. Defaults to True.

        Returns:
            np.ndarray: filtered signal
        """

        if updated:
            filtered = self.low_pass_filter_updated(signal)
            out = self.high_pass_filter_updated(filtered)
        else:
            filtered = self.low_pass_filter(signal)
            out = self.high_pass_filter(filtered)

        out = out / max(max(out), -min(out))

        return out

    def derivative(self, signal: np.ndarray, sample_rate: int = 100) -> np.ndarray:
        """ Original derivative filter.
            The derivative filter has the recursive equation:
            y(nT) = [-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]/(8T)

        Args:
            signal (np.ndarray): input signal
            sample_rate (int, optional): Sample rate. Defaults to 100.

        Returns:
            np.ndarray: Derivative of the signal
        """

        result = signal.copy()
        for index in range(len(signal)):
            result[index] = 0

            if (index >= 1):
                result[index] -= 2 * signal[index - 1]

            if (index >= 2):
                result[index] -= signal[index - 2]

            if (index >= 2 and index <= len(signal) - 2):
                result[index] += 2 * signal[index + 1]

            if (index >= 2 and index <= len(signal) - 3):
                result[index] += signal[index + 2]

            result[index] = (result[index] * sample_rate) / 8

        return result

    def derivative_updated(self, signal: np.ndarray) -> np.ndarray:
        """ Updated derivative filter.
            The derivative filter has the recursive equation:
            y(n) = 0.1 (-x(n - 2) - 2x(n - 1) + 2x(n + 1) + x(n + 2))

        Args:
            signal (np.ndarray): input signal

        Returns:
            np.ndarray: Derivative of the signal
        """

        filtered_signal = np.zeros_like(signal)

        for n in range(2, len(signal) - 2):
            filtered_signal[n] = 0.1 * (-signal[n - 2] - 2 * signal[n - 1] + 2 * signal[n + 1] + signal[n + 2])

        return filtered_signal

    def squaring(self, signal: np.ndarray) -> np.ndarray:
        """ Squares a signal.
            The squaring filter has the recursive equation:
            y(nT) = [x(nT)]^2

        Args:
            signal (np.ndarray): input signal

        Returns:
            np.ndarray: squared signal
        """

        result = np.zeros_like(signal)
        for index in range(len(signal)):
            result[index] = signal[index]**2

        return result

    def moving_window_integration(self,
                                  signal: np.ndarray,
                                  sample_rate: int = 100,
                                  window_size: float = 0.15) -> np.ndarray:
        """ Moving window integrator.
            The moving window integration has the recursive equation:
            y(nT) = [y(nT - (N-1)T) + x(nT - (N-2)T) + ... + x(nT)]/N
            N is the number of samples in the width of integration window.

        Args:
            signal (np.ndarray): input signal
            sample_rate (int, optional): Sample rate. Defaults to 100.
            window_size (float, optional): Window size as sample rate percentage. Defaults to 0.15.

        Returns:
            np.ndarray: processed signal
        """

        result = np.zeros_like(signal)
        win_size = round(window_size * sample_rate)
        sum = 0

        for j in range(win_size):
            sum += signal[j] / win_size
            result[j] = sum

        for index in range(win_size, len(signal)):
            sum += signal[index] / win_size
            sum -= signal[index - win_size] / win_size
            result[index] = sum

        return result

    def preprocess(self,
                   signal: np.ndarray,
                   updated: bool = True,
                   sample_rate: int = 100,
                   window_size: float = 0.15) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Pre-process for the Pan Tompkins peak detection

        Args:
            signal (np.ndarray): input signal
            updated (bool, optional): Use the updated version. Defaults to True.
            sample_rate (int, optional): Sample rate. Defaults to 100.
            window_size (float, optional): Window size as sample rate percentage. Defaults to 0.15.

        Returns:
            Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Band pass signal,
            derivative signal, squared signal and integrated signal.
        """

        band_pass_sgn = self.band_pass_filter(signal=signal, updated=updated)
        if updated:
            derivative_sgn = self.derivative_updated(signal=band_pass_sgn)
        else:
            derivative_sgn = self.derivative(signal=band_pass_sgn, sample_rate=sample_rate)

        sqr_sgn = self.squaring(signal=derivative_sgn)
        mov_win = self.moving_window_integration(signal=sqr_sgn, sample_rate=sample_rate, window_size=window_size)

        return band_pass_sgn, derivative_sgn, sqr_sgn, mov_win


class PanTompkinsQRS():

    def __init__(self, signal: np.ndarray, sample_rate: int = 100, window_size: float = 0.15) -> None:
        """ Initialize the class

        Args:
            signal (np.ndarray): input signal
            sample_rate (int, optional): Sample rate. Defaults to 100.
            window_size (float, optional): Window size as sample rate percentage. Defaults to 0.15.
        """

        self.sample_rate = sample_rate
        self.signal = signal
        self.window_size = round(window_size * self.sample_rate)
        preprocessor = PanTompkinsPreprocess()
        self.band_pass_sgn, _, _, self.mov_win_sgn = preprocessor.preprocess(signal=signal,
                                                                             updated=True,
                                                                             sample_rate=sample_rate,
                                                                             window_size=window_size)

    def reset(self) -> None:
        """ Reset the internal variables
        """
        # R peaks positions
        self.peaks = []
        # estimated position of possible peaks
        self.probable_peaks = []
        # signal level of the integrated signal (mov_win_sign)
        self.signal_lvl_I = 0
        # noise level of the integrated signal (mov_win_sign)
        self.noise_lvl_I = 0
        # signal level of the filtered signal (band_pass_sgn)
        self.signal_lvl_F = 0
        # noise level of the filtered signal (band_pass_sgn)
        self.noise_lvl_F = 0
        # threshold of the integrated signal that considers the noise level and previous detected QRS complex
        self.thres_I1 = 0
        # threshold of the filtered signal that considers the noise level and previous detected QRS complex
        self.thres_F1 = 0
        # lower threshold for the integrated signal
        self.thres_I2 = 0
        # lower threshold for the filtered signal
        self.thres_F2 = 0

        # R peak intervals
        self.rr1 = []
        # average of the last R peak intervals
        self.rr_avg1 = 0
        # threshold for low R peak interval
        self.rr_low_limit = 0
        # threshold for high R peak interval
        self.rr_high_limit = 0
        # R peak intervals between rr_low_limit and rr_high_limit
        self.rr2 = []
        # average of the last R peak intervals between limits
        self.rr_avg2 = 0
        # max interval without a peak detected
        self.rr_missed_limit = 0
        # is a T wave
        self.T_wave = False
        # R peaks locations
        self.r_peaks_loc = []

        # fine tuned R peaks
        self.tuned_peaks = []

    def estimate_r_peaks(self) -> None:
        """ Estimate the R peaks.
        """

        slopes = sg.fftconvolve(self.mov_win_sgn, np.full((25,), 1) / 25, mode='same')

        # approximate peak locations
        for i in range(round(0.5 * self.sample_rate) + 1, len(slopes) - 1):
            if slopes[i] > slopes[i - 1] and slopes[i + 1] < slopes[i]:
                self.peaks.append(i)

    def adjust_thresholds(self, peak_loc: float, peak_idx: int) -> None:
        """ Adjust the noise and signal thresholds in the initial phase

        Args:
            peak_val (float): peak location
            peak_idx (int): peak index
        """

        # peak found in the integrated signal
        peak_I = self.mov_win_sgn[peak_loc]
        # peak found in the filtered signal
        peak_F = self.band_pass_sgn[peak_idx]

        if peak_I >= self.thres_I1:
            # Update signal threshold
            self.signal_lvl_I = 0.125 * peak_I + 0.875 * self.signal_lvl_I

            if self.probable_peaks[peak_idx] > self.thres_F1:
                self.signal_lvl_F = 0.125 * peak_F + 0.875 * self.signal_lvl_F
                self.r_peaks_loc.append(self.probable_peaks[peak_idx])

            else:
                # Update noise threshold
                self.noise_lvl_F = 0.125 * peak_F + 0.875 * self.noise_lvl_F

        # Update noise thresholds
        elif peak_I < self.thres_I2 or self.thres_I2 < peak_I < self.thres_I1:
            self.noise_lvl_I = 0.125 * peak_I + 0.875 * self.noise_lvl_I
            self.noise_lvl_F = 0.125 * peak_F + 0.875 * self.noise_lvl_F

    def adjust_rr_interval(self, peak_idx: int) -> None:
        """ Adjust the R peaks intervals and limits

        Args:
            peak_idx (int): peak index
        """

        # find the n_intervals most recent R peak intervals
        n_intervals = 8
        self.rr1 = np.diff(self.peaks[max(0, peak_idx - n_intervals):peak_idx + 1]) / self.sample_rate

        self.rr_avg1 = np.mean(self.rr1)
        self.rr_avg2 = self.rr_avg1

        # find the n_intervals recent R peak intervals between R peak low and high limits
        if peak_idx >= n_intervals:
            for i in range(0, n_intervals):
                if self.rr_low_limit < self.rr1[i] < self.rr_high_limit:
                    self.rr2.append(self.rr1[i])

                    if len(self.rr2) > n_intervals:
                        self.rr2.remove(self.rr2[0])
                        self.rr_avg2 = np.mean(self.rr2)

        # update the R peak intervals low and high limits
        if len(self.rr2) > n_intervals - 1 or peak_idx < n_intervals:
            self.rr_low_limit = 0.92 * self.rr_avg2
            self.rr_high_limit = 1.16 * self.rr_avg2
            self.rr_missed_limit = 1.66 * self.rr_avg2

        # update thresholds for irregular beats
        if self.rr_avg1 < self.rr_low_limit or self.rr_avg1 > self.rr_missed_limit:
            self.thres_I1 /= 2
            self.thres_F1 /= 2

    def searchback_for_missed_qrs(self, peak_idx: int) -> None:
        """ Search back for missed QRS complexes.

        Args:
            peak_idx (int): peak location
        """

        # Check if the most recent RR interval is greater than the RR Missed Limit
        search_window_size = round(self.rr1[-1] * self.sample_rate)
        search_window = self.mov_win_sgn[peak_idx - search_window_size + 1:peak_idx + 1]

        # coordinates of points above threshold I1
        coord = np.where(search_window > self.thres_I1)

        # peak value in the search window
        x_max = np.argmax(search_window[coord]) if len(coord) > 0 else None

        if x_max is not None:
            # update the thresholds corresponding to integrated signal
            self.signal_lvl_I = 0.25 * self.mov_win_sgn[x_max] + 0.75 * self.signal_lvl_I
            self.thres_I1 = self.noise_lvl_I + 0.25 * (self.signal_lvl_I - self.noise_lvl_I)
            self.thres_I2 = 0.5 * self.thres_I1

            search_window = self.band_pass_sgn[x_max - self.window_size:min(len(self.band_pass_sgn) - 1, x_max)]
            # coordinates of points above threshold F1
            coord = np.where(search_window > self.thres_F1)

            # peak value in the search window
            r_max = np.argmax(search_window[coord]) if len(coord) > 0 else None

            if r_max is not None:
                # update the thresholds corresponding to filtered signal
                if self.band_pass_sgn[r_max] > self.thres_F2:
                    self.signal_lvl_F = 0.25 * self.band_pass_sgn[r_max] + 0.75 * self.signal_lvl_F
                    self.thres_F1 = self.noise_lvl_F + 0.25 * (self.signal_lvl_F - self.noise_lvl_F)
                    self.thres_F2 = 0.5 * self.thres_F1

                    # probable R peak location
                    self.r_peaks_loc.append(r_max)

    def t_wave_discrimination(self, peak_loc: int, last_rr: float, curr_idx: int, prev_idx: int) -> None:
        """ Identify if it is a T wave.

        Args:
            peak_loc (int): peak location
            last_rr (float): last R peak interval
            curr_idx (int): current peak index
            prev_idx (int): previous peak index
        """

        if self.mov_win_sgn[peak_loc] >= self.thres_I1:
            if curr_idx > 0 and 0.20 < last_rr < 0.36:
                # current and last slopes
                half_window = round(self.window_size / 2)
                curr_slope = max(np.diff(self.mov_win_sgn[peak_loc - half_window:peak_loc + 1]))
                last_slope = max(np.diff(self.mov_win_sgn[self.peaks[prev_idx] - half_window:self.peaks[prev_idx] + 1]))

                if curr_slope < 0.5 * last_slope:
                    self.T_wave = True
                    self.noise_lvl_I = 0.125 * self.mov_win_sgn[peak_loc] + 0.875 * self.noise_lvl_I

            if not self.T_wave:
                if self.probable_peaks[curr_idx] > self.thres_F1:
                    self.signal_lvl_I = 0.125 * self.mov_win_sgn[peak_loc] + 0.875 * self.signal_lvl_I
                    self.signal_lvl_F = 0.125 * self.band_pass_sgn[curr_idx] + 0.875 * self.signal_lvl_F
                    self.r_peaks_loc.append(self.probable_peaks[curr_idx])
                else:
                    self.signal_lvl_I = 0.125 * self.mov_win_sgn[peak_loc] + 0.875 * self.signal_lvl_I
                    self.noise_lvl_F = 0.125 * self.band_pass_sgn[curr_idx] + 0.875 * self.noise_lvl_F

        # update noise thresholds
        elif self.mov_win_sgn[peak_loc] < self.thres_I1 or \
            self.thres_I1 < self.mov_win_sgn[peak_loc] < self.thres_I2:
            self.noise_lvl_I = 0.125 * self.mov_win_sgn[peak_loc] + 0.875 * self.noise_lvl_I
            self.noise_lvl_F = 0.125 * self.band_pass_sgn[curr_idx] + 0.875 * self.noise_lvl_F

    def update_next_thresholds(self) -> None:
        """ Update the noise and signal thresholds for the next iteration.
        """

        self.thres_I1 = self.noise_lvl_I + 0.25 * (self.signal_lvl_I - self.noise_lvl_I)
        self.thres_F1 = self.noise_lvl_F + 0.25 * (self.signal_lvl_F - self.noise_lvl_F)
        self.thres_I2 = 0.5 * self.thres_I1
        self.thres_F2 = 0.5 * self.thres_F1
        self.T_wave = False

    def fine_tune_r_peaks(self, window_size: float = 0.2) -> np.ndarray:
        """ Search the peaks on the original signal for a more exact location.

        Args:
            window_size (float, optional): Window size as sample rate percentage.. Defaults to 0.2.

        Returns:
            np.ndarray: Fine tuned R peaks

        """
        self.r_peaks_loc = np.unique(np.array(self.r_peaks_loc).astype(int))
        search_window = round(window_size * self.sample_rate)

        for r_val in self.r_peaks_loc:
            coord = np.arange(r_val - search_window, min(len(self.signal), r_val + search_window + 1), 1)
            x_max = np.argmax(self.signal[coord]) if len(coord) > 0 else None

            if x_max is not None:
                self.tuned_peaks.append(coord[x_max])

    def find_r_peaks(self, fine_tune_window: float = 0.2) -> None:
        """ Find the R peaks of the signal.

        Args:
            fine_tune_window (float, optional): Search window to fine tune the peaks on the filtered signal . Defaults to 0.2.
        """

        self.reset()
        self.estimate_r_peaks()

        for idx in range(len(self.peaks)):
            peak_val = self.peaks[idx]
            large_search_window = np.arange(max(0, self.peaks[idx] - self.window_size),
                                            min(self.peaks[idx] + self.window_size,
                                                len(self.band_pass_sgn) - 1), 1)
            window_max_val = max(self.band_pass_sgn[large_search_window], default=0)

            if window_max_val != 0:
                x_coord = np.where(self.band_pass_sgn == window_max_val)
                self.probable_peaks.append(x_coord[0][0])

            if idx < len(self.probable_peaks) and idx != 0:
                self.adjust_rr_interval(idx)
                if self.rr1[-1] > self.rr_missed_limit:
                    self.searchback_for_missed_qrs(peak_val)

                # T Wave Identification
                self.t_wave_discrimination(peak_val, self.rr1[-1], idx, idx - 1)

            else:
                self.adjust_thresholds(peak_val, idx)

            self.update_next_thresholds()

        self.fine_tune_r_peaks(window_size=fine_tune_window)

        self.tuned_peaks = np.array(self.tuned_peaks)

    def estimate_heartrate(self) -> Union[float, float]:
        """ Estimate the heart rate based on the R peaks

        Returns:
            float: heart rate in BPM
            float: heart rate variability
        """

        heart_rate = (60 * self.sample_rate) / np.average(np.diff(self.tuned_peaks[1:]))

        heart_rate_var = np.std(self.rr1)

        return heart_rate, heart_rate_var
