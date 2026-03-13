// Author:  Liming Liu
// Date:    13-03-2026

// A basic software dead reckoning engine that translates IMU kinematics into
// both floating-point and fixed-point C++ loops.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kDt = 0.01;                 // 100 Hz
constexpr int kHz = 100;
constexpr int kPhaseSeconds = 2;             // accel, coast, decel are each 2 s
constexpr int kTotalSamples = kHz * kPhaseSeconds * 3;
constexpr double kAccelMagnitude = 1.0;      // m/s^2
constexpr double kNoiseStdDev = 0.05;        // m/s^2

constexpr int kFpShift = 10;                 // Q22.10 style scale factor
constexpr int32_t kScale = 1 << kFpShift;    // 1024

struct SampleResult {
	double time_s;
	double accel_true;
	double accel_noisy;
	double vel_float;
	double pos_float;
	double vel_fixed;
	double pos_fixed;
};

int32_t clampToInt32(int64_t value) {
	if (value > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
		return std::numeric_limits<int32_t>::max();
	}
	if (value < static_cast<int64_t>(std::numeric_limits<int32_t>::min())) {
		return std::numeric_limits<int32_t>::min();
	}
	return static_cast<int32_t>(value);
}

int32_t toFixed(double value) {
	const double scaled = value * static_cast<double>(kScale);
	return clampToInt32(static_cast<int64_t>(std::llround(scaled)));
}

double fromFixed(int32_t value) {
	return static_cast<double>(value) / static_cast<double>(kScale);
}

// Multiply two fixed-point values and immediately shift to keep scale stable.
int32_t mulFixed(int32_t a, int32_t b) {
	const int64_t wide = static_cast<int64_t>(a) * static_cast<int64_t>(b);
	const int64_t shifted = wide >> kFpShift;
	return clampToInt32(shifted);
}

void writeResults(const std::vector<SampleResult>& rows, const std::string& path) {
	std::ofstream out(path);
	if (!out) {
		throw std::runtime_error("Failed to open output file: " + path);
	}

	out << "time_s,accel_true,accel_noisy,vel_float,pos_float,vel_fixed,pos_fixed,"
		   "pos_error\n";
	out << std::fixed << std::setprecision(6);

	for (const auto& r : rows) {
		const double error = r.pos_fixed - r.pos_float;
		out << r.time_s << ',' << r.accel_true << ',' << r.accel_noisy << ','
			<< r.vel_float << ',' << r.pos_float << ',' << r.vel_fixed << ','
			<< r.pos_fixed << ',' << error << '\n';
	}
}

}  // namespace

int main() {
	try {
		// 1) Build deterministic dummy motion profile at 100 Hz.
		std::vector<double> accel_true(kTotalSamples, 0.0);
		for (int i = 0; i < kTotalSamples; ++i) {
			if (i < 2 * kHz) {
				accel_true[i] = kAccelMagnitude;      // accelerate 0-2 s
			} else if (i < 4 * kHz) {
				accel_true[i] = 0.0;                 // coast 2-4 s
			} else {
				accel_true[i] = -kAccelMagnitude;    // decelerate 4-6 s
			}
		}

		// 3) Inject Gaussian noise.
		std::mt19937 rng(42);  // fixed seed for reproducibility
		std::normal_distribution<double> noise_dist(0.0, kNoiseStdDev);

		std::vector<double> accel_noisy(kTotalSamples, 0.0);
		for (int i = 0; i < kTotalSamples; ++i) {
			accel_noisy[i] = accel_true[i] + noise_dist(rng);
		}

		// 2) Floating-point integration state.
		double vel_f = 0.0;
		double pos_f = 0.0;

		// 4/5) Fixed-point integration state (Q22.10 style).
		int32_t vel_fx = 0;
		int32_t pos_fx = 0;
		const int32_t dt_fx = toFixed(kDt);
		const int32_t half_fx = toFixed(0.5);

		std::vector<SampleResult> results;
		results.reserve(kTotalSamples);

		for (int i = 0; i < kTotalSamples; ++i) {
			const double a = accel_noisy[i];

			// Floating-point reference update.
			const double vel_next_f = vel_f + a * kDt;
			const double pos_next_f = pos_f + vel_f * kDt + 0.5 * a * kDt * kDt;

			// Fixed-point update; every multiplication passes through mulFixed.
			const int32_t a_fx = toFixed(a);
			const int32_t a_dt_fx = mulFixed(a_fx, dt_fx);             // a * dt
			const int32_t vel_dt_fx = mulFixed(vel_fx, dt_fx);         // v * dt
			const int32_t dt_sq_fx = mulFixed(dt_fx, dt_fx);           // dt^2
			const int32_t a_dt_sq_fx = mulFixed(a_fx, dt_sq_fx);       // a * dt^2
			const int32_t half_a_dt_sq_fx = mulFixed(half_fx, a_dt_sq_fx);  // 0.5*a*dt^2

			const int32_t vel_next_fx = clampToInt32(
				static_cast<int64_t>(vel_fx) + static_cast<int64_t>(a_dt_fx));
			const int32_t pos_next_fx = clampToInt32(
				static_cast<int64_t>(pos_fx) + static_cast<int64_t>(vel_dt_fx)
				+ static_cast<int64_t>(half_a_dt_sq_fx));

			vel_f = vel_next_f;
			pos_f = pos_next_f;
			vel_fx = vel_next_fx;
			pos_fx = pos_next_fx;

			results.push_back(SampleResult{
				static_cast<double>(i + 1) * kDt,
				accel_true[i],
				a,
				vel_f,
				pos_f,
				fromFixed(vel_fx),
				fromFixed(pos_fx)
			});
		}

		// 6) Save comparison output for plotting in Python.
		writeResults(results, "imu_dead_reckoning_results.csv");

		const auto& last = results.back();
		const double final_error = last.pos_fixed - last.pos_float;

		std::cout << std::fixed << std::setprecision(6);
		std::cout << "Wrote imu_dead_reckoning_results.csv with " << results.size()
				  << " samples\n";
		std::cout << "Final float position: " << last.pos_float << " m\n";
		std::cout << "Final fixed position: " << last.pos_fixed << " m\n";
		std::cout << "Final position error (fixed - float): " << final_error
				  << " m\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << '\n';
		return 1;
	}
}