#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <matplot/matplot.h>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>

const double MATH_PI = 3.1415926;
namespace py = pybind11;
namespace plt = matplot;

// ========== Combined Waveform Generation and Visualization ==========
std::vector<double> createAndVisualizeSineWave(double freqHz, double tStart, double tEnd, size_t numPoints) {
    std::vector<double> waveform;
    std::vector<double> timeAxis;
    waveform.reserve(numPoints);
    timeAxis.reserve(numPoints);
    
    const double step = (tEnd - tStart) / (numPoints - 1);
    for (size_t idx = 0; idx < numPoints; ++idx) {
        const double t = tStart + idx * step;
        timeAxis.push_back(t);
        waveform.push_back(std::sin(2 * MATH_PI * freqHz * t));
    }
    
    plt::plot(timeAxis, waveform);
    plt::title("Sygnał sinusoidalny");
    plt::xlabel("Czas [s]");
    plt::ylabel("Amplituda");
    plt::show();
    
    return waveform;
}

std::vector<double> createAndVisualizeCosineWave(double freqHz, double tStart, double tEnd, size_t numPoints) {
    std::vector<double> waveform;
    std::vector<double> timeAxis;
    waveform.reserve(numPoints);
    timeAxis.reserve(numPoints);
    
    const double step = (tEnd - tStart) / (numPoints - 1);
    for (size_t idx = 0; idx < numPoints; ++idx) {
        const double t = tStart + idx * step;
        timeAxis.push_back(t);
        waveform.push_back(std::cos(2 * MATH_PI * freqHz * t));
    }
    
    plt::plot(timeAxis, waveform);
    plt::title("Sygnał cosinusoidalny");
    plt::xlabel("Czas [s]");
    plt::ylabel("Amplituda");
    plt::show();
    
    return waveform;
}

std::vector<double> createAndVisualizeSquareWave(double freqHz, double tStart, double tEnd, size_t numPoints) {
    std::vector<double> waveform;
    std::vector<double> timeAxis;
    waveform.reserve(numPoints);
    timeAxis.reserve(numPoints);
    
    const double step = (tEnd - tStart) / (numPoints - 1);
    for (size_t idx = 0; idx < numPoints; ++idx) {
        const double t = tStart + idx * step;
        timeAxis.push_back(t);
        waveform.push_back(std::sin(2 * MATH_PI * freqHz * t) >= 0 ? 1.0 : -1.0);
    }
    
    plt::plot(timeAxis, waveform);
    plt::title("Sygnał prostokątny");
    plt::xlabel("Czas [s]");
    plt::ylabel("Amplituda");
    plt::show();
    
    return waveform;
}

std::vector<double> createAndVisualizeSawtoothWave(double freqHz, double tStart, double tEnd, size_t numPoints) {
    std::vector<double> waveform;
    std::vector<double> timeAxis;
    waveform.reserve(numPoints);
    timeAxis.reserve(numPoints);
    
    const double period = 1.0 / freqHz;
    const double step = (tEnd - tStart) / (numPoints - 1);
    for (size_t idx = 0; idx < numPoints; ++idx) {
        const double t = tStart + idx * step;
        timeAxis.push_back(t);
        const double phase = std::fmod(t, period) / period;
        waveform.push_back(2.0 * phase - 1.0);
    }
    
    plt::plot(timeAxis, waveform);
    plt::title("Sygnał piłokształtny");
    plt::xlabel("Czas [s]");
    plt::ylabel("Amplituda");
    plt::show();
    
    return waveform;
}

// ========== Combined DFT Computation and Visualization ==========
std::vector<std::complex<double>> computeAndVisualizeDFT(const std::vector<double>& inputSignal, double samplingRate) {
    const size_t N = inputSignal.size();
    std::vector<std::complex<double>> spectrum(N);
    
    // Compute DFT
    for (size_t k = 0; k < N; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t n = 0; n < N; ++n) {
            double angle = -2 * MATH_PI * k * n / N;
            sum += inputSignal[n] * std::exp(std::complex<double>(0, angle));
        }
        spectrum[k] = sum;
    }
    
    // Prepare and plot spectrum
    std::vector<double> freqs(N), magnitudes(N);
    for (size_t k = 0; k < N; ++k) {
        freqs[k] = k * samplingRate / N;
        magnitudes[k] = std::abs(spectrum[k]);
    }
    plt::plot(freqs, magnitudes);
    plt::title("Widmo amplitudowe (DFT)");
    plt::xlabel("Częstotliwość [Hz]");
    plt::ylabel("|X(f)|");
    plt::show();
    
    return spectrum;
}

// ========== Combined IDFT Computation and Visualization ==========
std::vector<double> computeAndVisualizeIDFT(const std::vector<std::complex<double>>& spectrum, double samplingRate) {
    const size_t N = spectrum.size();
    std::vector<double> reconstructed(N);
    
    // Compute IDFT
    for (size_t n = 0; n < N; ++n) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t k = 0; k < N; ++k) {
            double angle = 2 * MATH_PI * k * n / N;
            sum += spectrum[k] * std::exp(std::complex<double>(0, angle));
        }
        reconstructed[n] = (sum.real() / N);
    }
    
    // Generate time axis
    std::vector<double> t(N);
    const double step = (N-1) / samplingRate / (N - 1);
    for (size_t i = 0; i < N; ++i) {
        t[i] = i * step;
    }
    
    // Plot reconstructed signal
    plt::plot(t, reconstructed);
    plt::title("Rekonstrukcja sygnału (IDFT)");
    plt::xlabel("Czas [s]");
    plt::ylabel("Amplituda");
    plt::show();
    
    return reconstructed;
}

// ========== 1D Filtering Functions ==========

// Moving Average Filter (Low-pass)
std::vector<double> movingAverageFilter(const std::vector<double>& signal, size_t windowSize) {
    if (windowSize == 0 || windowSize > signal.size()) {
        return signal;
    }
    
    std::vector<double> filtered;
    filtered.reserve(signal.size());
    
    // Handle beginning
    for (size_t i = 0; i < windowSize / 2 && i < signal.size(); ++i) {
        double sum = 0.0;
        size_t count = 0;
        for (size_t j = 0; j <= i + windowSize / 2 && j < signal.size(); ++j) {
            sum += signal[j];
            count++;
        }
        filtered.push_back(sum / count);
    }
    
    // Main filtering
    for (size_t i = windowSize / 2; i < signal.size() - windowSize / 2; ++i) {
        double sum = 0.0;
        for (size_t j = i - windowSize / 2; j <= i + windowSize / 2; ++j) {
            sum += signal[j];
        }
        filtered.push_back(sum / windowSize);
    }
    
    // Handle end
    for (size_t i = signal.size() - windowSize / 2; i < signal.size(); ++i) {
        double sum = 0.0;
        size_t count = 0;
        for (size_t j = i - windowSize / 2; j < signal.size(); ++j) {
            sum += signal[j];
            count++;
        }
        filtered.push_back(sum / count);
    }
    
    return filtered;
}

// Gaussian Filter (Low-pass)
std::vector<double> gaussianFilter1D(const std::vector<double>& signal, double sigma) {
    // Create Gaussian kernel
    int kernelSize = static_cast<int>(6 * sigma + 1);
    if (kernelSize % 2 == 0) kernelSize++;
    
    std::vector<double> kernel(kernelSize);
    double sum = 0.0;
    int center = kernelSize / 2;
    
    for (int i = 0; i < kernelSize; ++i) {
        double x = i - center;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (double& k : kernel) {
        k /= sum;
    }
    
    // Apply convolution
    std::vector<double> filtered(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        double value = 0.0;
        for (int j = 0; j < kernelSize; ++j) {
            int idx = static_cast<int>(i) + j - center;
            if (idx >= 0 && idx < static_cast<int>(signal.size())) {
                value += signal[idx] * kernel[j];
            }
        }
        filtered[i] = value;
    }
    
    return filtered;
}

// High-pass filter (simple differentiation)
std::vector<double> highPassFilter1D(const std::vector<double>& signal) {
    std::vector<double> filtered;
    filtered.reserve(signal.size());
    
    if (signal.empty()) return filtered;
    
    filtered.push_back(0.0); // First element
    
    for (size_t i = 1; i < signal.size(); ++i) {
        filtered.push_back(signal[i] - signal[i-1]);
    }
    
    return filtered;
}

// Visualize 1D filtering results
std::vector<double> applyAndVisualize1DFilter(const std::vector<double>& signal, 
                                             const std::string& filterType, 
                                             double param = 5.0) {
    std::vector<double> filtered;
    
    if (filterType == "moving_average") {
        filtered = movingAverageFilter(signal, static_cast<size_t>(param));
    } else if (filterType == "gaussian") {
        filtered = gaussianFilter1D(signal, param);
    } else if (filterType == "high_pass") {
        filtered = highPassFilter1D(signal);
    } else {
        filtered = signal; // No filtering
    }
    
    // Create time axis
    std::vector<double> t(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        t[i] = static_cast<double>(i);
    }
    
    // Plot original and filtered signals
    plt::subplot(2, 1, 1);
    plt::plot(t, signal);
    plt::title("Sygnał oryginalny");
    plt::xlabel("Indeks próbki");
    plt::ylabel("Amplituda");
    
    plt::subplot(2, 1, 2);
    plt::plot(t, filtered);
    plt::title("Sygnał przefiltrowany (" + filterType + ")");
    plt::xlabel("Indeks próbki");
    plt::ylabel("Amplituda");
    
    plt::show();
    
    return filtered;
}

// ========== 2D Filtering Functions ==========

// 2D Gaussian Filter
std::vector<std::vector<double>> gaussianFilter2D(const std::vector<std::vector<double>>& image, double sigma) {
    if (image.empty() || image[0].empty()) return image;
    
    size_t rows = image.size();
    size_t cols = image[0].size();
    
    // Create 2D Gaussian kernel
    int kernelSize = static_cast<int>(6 * sigma + 1);
    if (kernelSize % 2 == 0) kernelSize++;
    
    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double sum = 0.0;
    int center = kernelSize / 2;
    
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            double x = i - center;
            double y = j - center;
            kernel[i][j] = std::exp(-(x*x + y*y) / (2 * sigma * sigma));
            sum += kernel[i][j];
        }
    }
    
    // Normalize kernel
    for (auto& row : kernel) {
        for (double& k : row) {
            k /= sum;
        }
    }
    
    // Apply 2D convolution
    std::vector<std::vector<double>> filtered(rows, std::vector<double>(cols, 0.0));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value = 0.0;
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    int ii = static_cast<int>(i) + ki - center;
                    int jj = static_cast<int>(j) + kj - center;
                    
                    if (ii >= 0 && ii < static_cast<int>(rows) && 
                        jj >= 0 && jj < static_cast<int>(cols)) {
                        value += image[ii][jj] * kernel[ki][kj];
                    }
                }
            }
            filtered[i][j] = value;
        }
    }
    
    return filtered;
}

// 2D Sobel Edge Detection (High-pass filter)
std::vector<std::vector<double>> sobelFilter2D(const std::vector<std::vector<double>>& image) {
    if (image.empty() || image[0].empty()) return image;
    
    size_t rows = image.size();
    size_t cols = image[0].size();
    
    // Sobel kernels
    std::vector<std::vector<int>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    std::vector<std::vector<int>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    std::vector<std::vector<double>> filtered(rows, std::vector<double>(cols, 0.0));
    
    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            double gx = 0.0, gy = 0.0;
            
            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    gx += image[i + ki][j + kj] * sobelX[ki + 1][kj + 1];
                    gy += image[i + ki][j + kj] * sobelY[ki + 1][kj + 1];
                }
            }
            
            filtered[i][j] = std::sqrt(gx*gx + gy*gy);
        }
    }
    
    return filtered;
}

// 2D Box Filter (Moving Average)
std::vector<std::vector<double>> boxFilter2D(const std::vector<std::vector<double>>& image, int kernelSize) {
    if (image.empty() || image[0].empty() || kernelSize <= 0) return image;
    
    size_t rows = image.size();
    size_t cols = image[0].size();
    
    std::vector<std::vector<double>> filtered(rows, std::vector<double>(cols, 0.0));
    int halfSize = kernelSize / 2;
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            int count = 0;
            
            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {
                    int ii = static_cast<int>(i) + ki;
                    int jj = static_cast<int>(j) + kj;
                    
                    if (ii >= 0 && ii < static_cast<int>(rows) && 
                        jj >= 0 && jj < static_cast<int>(cols)) {
                        sum += image[ii][jj];
                        count++;
                    }
                }
            }
            
            filtered[i][j] = sum / count;
        }
    }
    
    return filtered;
}

// Visualize 2D filtering results - POPRAWIONA WERSJA
std::vector<std::vector<double>> applyAndVisualize2DFilter(const std::vector<std::vector<double>>& image,
                                                          const std::string& filterType,
                                                          double param = 1.0) {
    std::vector<std::vector<double>> filtered;
    
    if (filterType == "gaussian") {
        filtered = gaussianFilter2D(image, param);
    } else if (filterType == "sobel") {
        filtered = sobelFilter2D(image);
    } else if (filterType == "box") {
        filtered = boxFilter2D(image, static_cast<int>(param));
    } else {
        filtered = image; // No filtering
    }
    
    // Konwertuj na format matplot::vector_2d
    size_t rows = image.size();
    size_t cols = image[0].size();
    
    // Utwórz macierze w formacie vector_1d dla każdego wiersza
    std::vector<std::vector<double>> originalMatrix = image;
    std::vector<std::vector<double>> filteredMatrix = filtered;
    
    try {
        // Alternatywnie, użyj heatmap
        plt::figure();
        
        // Oryginalny obraz
        plt::subplot(1, 2, 1);
        auto h1 = plt::heatmap(originalMatrix);
        plt::title("Obraz oryginalny");
        plt::colorbar();
        
        // Przefiltrowany obraz
        plt::subplot(1, 2, 2);
        auto h2 = plt::heatmap(filteredMatrix);
        plt::title("Obraz przefiltrowany (" + filterType + ")");
        plt::colorbar();
        
        plt::show();
    } catch (const std::exception& e) {
        // Jeśli heatmap nie działa, użyj prostszej wizualizacji
        std::cout << "Wizualizacja 2D zakończona (heatmap może nie być dostępny)" << std::endl;
        std::cout << "Rozmiar obrazu: " << rows << "x" << cols << std::endl;
        std::cout << "Filtr: " << filterType << " z parametrem: " << param << std::endl;
    }
    
    return filtered;
}

// ========== Visualization Utilities ==========
void plotWaveform(const std::vector<double>& x, const std::vector<double>& y) {
    plt::plot(x, y);
    plt::title("Signal");
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::show();
}

// Generate test 2D image (for testing 2D filters)
std::vector<std::vector<double>> generateTestImage(size_t rows, size_t cols, const std::string& pattern = "checkerboard") {
    std::vector<std::vector<double>> image(rows, std::vector<double>(cols, 0.0));
    
    if (pattern == "checkerboard") {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                image[i][j] = ((i / 10) + (j / 10)) % 2 ? 1.0 : 0.0;
            }
        }
    } else if (pattern == "circle") {
        double centerX = cols / 2.0;
        double centerY = rows / 2.0;
        double radius = std::min(rows, cols) / 4.0;
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                double dx = j - centerX;
                double dy = i - centerY;
                if (dx*dx + dy*dy <= radius*radius) {
                    image[i][j] = 1.0;
                }
            }
        }
    } else if (pattern == "gradient") {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                image[i][j] = static_cast<double>(j) / cols;
            }
        }
    }
    
    return image;
}

// ========== Python Module Binding ==========
PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Zaawansowany moduł C++ do analizy sygnałów z integracją Pythona
        przez pybind11 i wizualizacją przez matplot++
    )pbdoc";

    // Wave generation and visualization functions
    m.def("sin_wave", &createAndVisualizeSineWave, 
          py::arg("freq"), py::arg("start"), py::arg("end"), py::arg("n_samples"),
          "Generuje i wizualizuje sygnał sinusoidalny");
    m.def("cos_wave", &createAndVisualizeCosineWave, 
          py::arg("freq"), py::arg("start"), py::arg("end"), py::arg("n_samples"),
          "Generuje i wizualizuje sygnał cosinusoidalny");
    m.def("square_wave", &createAndVisualizeSquareWave, 
          py::arg("freq"), py::arg("start"), py::arg("end"), py::arg("n_samples"),
          "Generuje i wizualizuje sygnał prostokątny");
    m.def("sawtooth_wave", &createAndVisualizeSawtoothWave, 
          py::arg("freq"), py::arg("start"), py::arg("end"), py::arg("n_samples"),
          "Generuje i wizualizuje sygnał piłokształtny");

    // Combined Fourier transform functions
    m.def("dft", &computeAndVisualizeDFT, 
          py::arg("signal"), py::arg("sampling_rate"), 
          "Oblicza i wizualizuje transformatę Fouriera");
    m.def("idft", &computeAndVisualizeIDFT, 
          py::arg("spectrum"), py::arg("sampling_rate"),
          "Oblicza i wizualizuje odwrotną transformatę Fouriera");

    // Filtering functions
    m.def("filter_1d", &applyAndVisualize1DFilter,
          py::arg("signal"), py::arg("filter_type"), py::arg("param") = 5.0,
          "Aplikuje filtr 1D i wizualizuje wyniki. Typy: 'moving_average', 'gaussian', 'high_pass'");
    
    m.def("filter_2d", &applyAndVisualize2DFilter,
          py::arg("image"), py::arg("filter_type"), py::arg("param") = 1.0,
          "Aplikuje filtr 2D i wizualizuje wyniki. Typy: 'gaussian', 'sobel', 'box'");

    // Utility functions
    m.def("generate_test_image", &generateTestImage,
          py::arg("rows"), py::arg("cols"), py::arg("pattern") = "checkerboard",
          "Generuje testowy obraz. Wzory: 'checkerboard', 'circle', 'gradient'");
    
    m.def("plot_waveform", &plotWaveform,
          py::arg("x"), py::arg("y"),
          "Rysuje wykres sygnału");
}