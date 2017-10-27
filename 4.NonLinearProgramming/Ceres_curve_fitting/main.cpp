// Use Ceres to fit the function: y = exp(ax^2 + bx + c) + w, w is noise, find optimized a, b, c
#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

struct CURVE_FITTING_COST {
	CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

	// for ceres use and define different cost function
	template <typename T>
	bool operator() (const T* const abc, T* residual) const {
		residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
		return true;
	}

	const double _x, _y;
};

int main(int argc, char const *argv[])
{
	double a = 1.0, b = 2.0, c = 1.0; // real parameters
	int N = 100; // number of samples
	double w_sigma = 1.0; // noise sigma
	cv::RNG rng; // OPenCV random number generator(for noise)
	double abc[3] = {0, 0, 0};

	vector<double> x_data, y_data;

	cout << "Generating data: " << endl;
	for (int i = 0; i < N; ++i) {
		double x = i / 100.0;
		x_data.push_back(x);
		y_data.push_back(exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
		cout << x_data[i] << "\t" << y_data[i] << endl;
	}

	// build least square problem using ceres
	ceres::Problem problem;
	for (int i = 0; i < N; ++i) {
		// <cost function, output dims, input dims>
		problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])), 
			nullptr, 
			abc
		);
	}

	// configure the solver
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	ceres::Solve(options, &problem, &summary);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "solve time cost = " << time_used.count() << " seconds." << endl;

	// output results
	cout << summary.BriefReport() << endl;
	cout << "estimated a, b, c = ";
	for (auto val : abc) {
		cout << val << " ";
	}
	cout << endl;

	return 0;
}