#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

// Physics Constants
const double G = 9.81;  // Acceleration due to gravity (m/s^2)
const double L1 = 1.0;  // Length of rod 1 (meters)
const double L2 = 1.0;  // Length of rod 2 (meters)
const double M1 = 1.0;  // Mass of pendulum 1 (kg)
const double M2 = 1.0;  // Mass of pendulum 2 (kg)

int main() {
    // Initial states: angles (theta) and angular velocities (omega)
    double t1 = 2.0;    // Angle of pendulum 1 (radians)
    double t2 = 1.5;    // Angle of pendulum 2 (radians)
    double w1 = 0.0;    // Angular velocity 1
    double w2 = 0.0;    // Angular velocity 2

    double dt = 0.01;       // Time step (seconds)
    int total_steps = 10000; // Generate 10,000 data rows for the ML model

    std::ofstream data_file("chaos_training_data.txt");
    if (!data_file.is_open()) {
        std::cerr << "File system access failure.\n";
        return 1;
    }

    std::cout << "Generating 10,000 steps of chaotic physical data...\n";

    for (int step = 0; step < total_steps; step++) {
        // Run continuous data step to solve the chaotic differential matrices
        double delta = t1 - t2;

        // Complicated denominators derived from Lagrangian mechanics
        double den1 = L1 * (2 * M1 + M2 - M2 * std::cos(2 * t1 - 2 * t2));
        double num1 = -G * (2 * M1 + M2) * std::sin(t1) - M2 * G * std::sin(t1 - 2 * t2) - 2 * std::sin(delta) * M2 * (w2 * w2 * L2 + w1 * w1 * L1 * std::cos(delta));
        double alpha1 = num1 / den1; // Angular acceleration 1

        double den2 = L2 * (2 * M1 + M2 - M2 * std::cos(2 * t1 - 2 * t2));
        double num2 = 2 * std::sin(delta) * (w1 * w1 * L1 * (M1 + M2) + G * (M1 + M2) * std::cos(t1) + w2 * w2 * L2 * M2 * std::cos(delta));
        double alpha2 = num2 / den2; // Angular acceleration 2

        // Euler-Cromer Integration Core
        w1 += alpha1 * dt;
        w2 += alpha2 * dt;
        t1 += w1 * dt;
        t2 += w2 * dt;

        // Save states sequentially to act as neural network training attributes
        data_file << t1 << " " << t2 << " " << w1 << " " << w2 << "\n";
    }

    data_file.close();
    std::cout << "Data generation complete. Matrix logged to chaos_training_data.txt.\n";
    return 0;
}
