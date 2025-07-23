/**
 * @file plotting.hpp
 * @brief Plotting utilities and hooks for ReservoirCpp
 * 
 * This module provides plotting utilities and an ABI for visualization
 * that can be implemented using various backends (matplotlib-cpp, etc.)
 * or exported to Python for visualization.
 * 
 * @author ReservoirCpp Development Team
 * @date 2024
 * @license MIT License
 */

#ifndef RESERVOIRCPP_PLOTTING_HPP
#define RESERVOIRCPP_PLOTTING_HPP

#include "reservoircpp/types.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace reservoircpp {
namespace plotting {

/**
 * @brief Plot configuration structure
 */
struct PlotConfig {
    std::string title;
    std::string xlabel;
    std::string ylabel;
    std::string color;
    std::string linestyle;
    std::string marker;
    float linewidth = 1.0f;
    float markersize = 3.0f;
    bool grid = true;
    bool legend = true;
    
    PlotConfig() : color("blue"), linestyle("-"), marker(""), 
                  linewidth(1.0f), markersize(3.0f), grid(true), legend(true) {}
};

/**
 * @brief Abstract base class for plotting backends
 */
class PlotBackend {
public:
    virtual ~PlotBackend() = default;

    /**
     * @brief Plot line chart
     * @param x X-axis data
     * @param y Y-axis data
     * @param config Plot configuration
     * @param label Data series label
     */
    virtual void plot_line(const Vector& x, const Vector& y, 
                          const PlotConfig& config = PlotConfig(),
                          const std::string& label = "") = 0;

    /**
     * @brief Plot scatter chart
     * @param x X-axis data
     * @param y Y-axis data
     * @param config Plot configuration
     * @param label Data series label
     */
    virtual void plot_scatter(const Vector& x, const Vector& y,
                             const PlotConfig& config = PlotConfig(),
                             const std::string& label = "") = 0;

    /**
     * @brief Plot heatmap
     * @param data 2D data matrix
     * @param config Plot configuration
     */
    virtual void plot_heatmap(const Matrix& data,
                             const PlotConfig& config = PlotConfig()) = 0;

    /**
     * @brief Plot histogram
     * @param data Data vector
     * @param bins Number of bins
     * @param config Plot configuration
     */
    virtual void plot_histogram(const Vector& data, int bins = 50,
                               const PlotConfig& config = PlotConfig()) = 0;

    /**
     * @brief Save current plot
     * @param filename Output filename
     * @param dpi Resolution in DPI
     */
    virtual void save_plot(const std::string& filename, int dpi = 300) = 0;

    /**
     * @brief Show plot (if supported by backend)
     */
    virtual void show_plot() = 0;

    /**
     * @brief Clear current plot
     */
    virtual void clear_plot() = 0;

    /**
     * @brief Create subplot
     * @param rows Number of subplot rows
     * @param cols Number of subplot columns
     * @param index Subplot index (1-based)
     */
    virtual void subplot(int rows, int cols, int index) = 0;
};

/**
 * @brief Python export backend - exports data for Python matplotlib
 */
class PythonExportBackend : public PlotBackend {
public:
    /**
     * @brief Constructor
     * @param output_dir Directory to save exported data
     */
    explicit PythonExportBackend(const std::string& output_dir = "plots");

    void plot_line(const Vector& x, const Vector& y, 
                  const PlotConfig& config = PlotConfig(),
                  const std::string& label = "") override;

    void plot_scatter(const Vector& x, const Vector& y,
                     const PlotConfig& config = PlotConfig(),
                     const std::string& label = "") override;

    void plot_heatmap(const Matrix& data,
                     const PlotConfig& config = PlotConfig()) override;

    void plot_histogram(const Vector& data, int bins = 50,
                       const PlotConfig& config = PlotConfig()) override;

    void save_plot(const std::string& filename, int dpi = 300) override;
    void show_plot() override;
    void clear_plot() override;
    void subplot(int rows, int cols, int index) override;

    /**
     * @brief Generate Python script to recreate plots
     * @param script_filename Output Python script filename
     */
    void generate_python_script(const std::string& script_filename = "plot_script.py");

private:
    std::string output_dir_;
    int plot_counter_;
    std::vector<std::string> plot_commands_;

    void export_vector(const Vector& data, const std::string& filename);
    void export_matrix(const Matrix& data, const std::string& filename);
    std::string config_to_python_args(const PlotConfig& config, const std::string& label);
};

/**
 * @brief Null backend - does nothing (for testing/headless environments)
 */
class NullBackend : public PlotBackend {
public:
    void plot_line(const Vector& x, const Vector& y, 
                  const PlotConfig& config = PlotConfig(),
                  const std::string& label = "") override {}

    void plot_scatter(const Vector& x, const Vector& y,
                     const PlotConfig& config = PlotConfig(),
                     const std::string& label = "") override {}

    void plot_heatmap(const Matrix& data,
                     const PlotConfig& config = PlotConfig()) override {}

    void plot_histogram(const Vector& data, int bins = 50,
                       const PlotConfig& config = PlotConfig()) override {}

    void save_plot(const std::string& filename, int dpi = 300) override {}
    void show_plot() override {}
    void clear_plot() override {}
    void subplot(int rows, int cols, int index) override {}
};

/**
 * @brief Main plotting interface
 */
class Plotter {
public:
    /**
     * @brief Constructor with backend
     * @param backend Plotting backend to use
     */
    explicit Plotter(std::unique_ptr<PlotBackend> backend);

    /**
     * @brief Set plotting backend
     * @param backend New backend to use
     */
    void set_backend(std::unique_ptr<PlotBackend> backend);

    /**
     * @brief Get current backend
     */
    PlotBackend& get_backend() { return *backend_; }

    // Convenience methods that delegate to backend
    void plot(const Vector& x, const Vector& y, 
              const PlotConfig& config = PlotConfig(),
              const std::string& label = "");

    void scatter(const Vector& x, const Vector& y,
                 const PlotConfig& config = PlotConfig(),
                 const std::string& label = "");

    void heatmap(const Matrix& data,
                 const PlotConfig& config = PlotConfig());

    void histogram(const Vector& data, int bins = 50,
                   const PlotConfig& config = PlotConfig());

    void save(const std::string& filename, int dpi = 300);
    void show();
    void clear();
    void subplot(int rows, int cols, int index);

    /**
     * @brief Plot time series
     * @param data Time series data
     * @param config Plot configuration
     */
    void plot_timeseries(const Matrix& data,
                        const PlotConfig& config = PlotConfig());

    /**
     * @brief Plot reservoir states over time
     * @param states Reservoir state matrix (time x units)
     * @param config Plot configuration
     */
    void plot_reservoir_states(const Matrix& states,
                              const PlotConfig& config = PlotConfig());

    /**
     * @brief Plot weight matrix
     * @param weights Weight matrix
     * @param config Plot configuration
     */
    void plot_weight_matrix(const Matrix& weights,
                           const PlotConfig& config = PlotConfig());

    /**
     * @brief Plot training/validation loss
     * @param train_loss Training loss over time
     * @param val_loss Validation loss over time (optional)
     * @param config Plot configuration
     */
    void plot_training_loss(const Vector& train_loss,
                           const Vector& val_loss = Vector(),
                           const PlotConfig& config = PlotConfig());

private:
    std::unique_ptr<PlotBackend> backend_;
};

/**
 * @brief Global plotting utilities
 */
class PlotUtils {
public:
    /**
     * @brief Get default plotter instance
     */
    static Plotter& get_default_plotter();

    /**
     * @brief Set default backend type
     * @param backend_type "python_export", "null", or custom
     * @param args Backend-specific arguments
     */
    static void set_default_backend(const std::string& backend_type,
                                   const std::string& args = "");

    /**
     * @brief Create backend from type string
     * @param backend_type Backend type
     * @param args Backend arguments
     * @return Created backend
     */
    static std::unique_ptr<PlotBackend> create_backend(const std::string& backend_type,
                                                      const std::string& args = "");

    /**
     * @brief Quick plot function
     * @param y Y-axis data
     * @param config Plot configuration
     */
    static void quick_plot(const Vector& y, const PlotConfig& config = PlotConfig());

    /**
     * @brief Quick scatter plot
     * @param x X-axis data
     * @param y Y-axis data
     * @param config Plot configuration
     */
    static void quick_scatter(const Vector& x, const Vector& y,
                             const PlotConfig& config = PlotConfig());

private:
    static std::unique_ptr<Plotter> default_plotter_;
    static std::string default_backend_type_;
    static std::string default_backend_args_;
};

} // namespace plotting
} // namespace reservoircpp

#endif // RESERVOIRCPP_PLOTTING_HPP