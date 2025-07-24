/**
 * @file plotting.cpp
 * @brief Implementation of plotting utilities
 */

#include "reservoircpp/plotting.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sstream>

namespace reservoircpp {
namespace plotting {

// PythonExportBackend Implementation
PythonExportBackend::PythonExportBackend(const std::string& output_dir)
    : output_dir_(output_dir), plot_counter_(0) {
    // Create directory if it doesn't exist (manual implementation for compatibility)
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());
}

void PythonExportBackend::plot_line(const Vector& x, const Vector& y, 
                                   const PlotConfig& config, const std::string& label) {
    std::string x_file = output_dir_ + "/x_data_" + std::to_string(plot_counter_) + ".csv";
    std::string y_file = output_dir_ + "/y_data_" + std::to_string(plot_counter_) + ".csv";
    
    export_vector(x, x_file);
    export_vector(y, y_file);
    
    std::string args = config_to_python_args(config, label);
    std::string command = "plt.plot(np.loadtxt('" + x_file + "'), np.loadtxt('" + 
                         y_file + "')" + args + ")";
    
    plot_commands_.push_back(command);
    plot_counter_++;
}

void PythonExportBackend::plot_scatter(const Vector& x, const Vector& y,
                                      const PlotConfig& config, const std::string& label) {
    std::string x_file = output_dir_ + "/x_scatter_" + std::to_string(plot_counter_) + ".csv";
    std::string y_file = output_dir_ + "/y_scatter_" + std::to_string(plot_counter_) + ".csv";
    
    export_vector(x, x_file);
    export_vector(y, y_file);
    
    std::string args = config_to_python_args(config, label);
    std::string command = "plt.scatter(np.loadtxt('" + x_file + "'), np.loadtxt('" + 
                         y_file + "')" + args + ")";
    
    plot_commands_.push_back(command);
    plot_counter_++;
}

void PythonExportBackend::plot_heatmap(const Matrix& data, const PlotConfig& config) {
    std::string data_file = output_dir_ + "/heatmap_data_" + std::to_string(plot_counter_) + ".csv";
    export_matrix(data, data_file);
    
    std::string command = "plt.imshow(np.loadtxt('" + data_file + "', delimiter=','), " +
                         "cmap='viridis', aspect='auto')";
    
    plot_commands_.push_back(command);
    plot_commands_.push_back("plt.colorbar()");
    
    if (!config.title.empty()) {
        plot_commands_.push_back("plt.title('" + config.title + "')");
    }
    
    plot_counter_++;
}

void PythonExportBackend::plot_histogram(const Vector& data, int bins, const PlotConfig& config) {
    std::string data_file = output_dir_ + "/hist_data_" + std::to_string(plot_counter_) + ".csv";
    export_vector(data, data_file);
    
    std::string command = "plt.hist(np.loadtxt('" + data_file + "'), bins=" + 
                         std::to_string(bins) + ", alpha=0.7)";
    
    plot_commands_.push_back(command);
    
    if (!config.title.empty()) {
        plot_commands_.push_back("plt.title('" + config.title + "')");
    }
    if (!config.xlabel.empty()) {
        plot_commands_.push_back("plt.xlabel('" + config.xlabel + "')");
    }
    if (!config.ylabel.empty()) {
        plot_commands_.push_back("plt.ylabel('" + config.ylabel + "')");
    }
    
    plot_counter_++;
}

void PythonExportBackend::save_plot(const std::string& filename, int dpi) {
    std::string command = "plt.savefig('" + filename + "', dpi=" + std::to_string(dpi) + ")";
    plot_commands_.push_back(command);
}

void PythonExportBackend::show_plot() {
    plot_commands_.push_back("plt.show()");
}

void PythonExportBackend::clear_plot() {
    plot_commands_.push_back("plt.clf()");
}

void PythonExportBackend::subplot(int rows, int cols, int index) {
    std::string command = "plt.subplot(" + std::to_string(rows) + ", " + 
                         std::to_string(cols) + ", " + std::to_string(index) + ")";
    plot_commands_.push_back(command);
}

void PythonExportBackend::generate_python_script(const std::string& script_filename) {
    std::ofstream script(output_dir_ + "/" + script_filename);
    
    script << "import numpy as np\n";
    script << "import matplotlib.pyplot as plt\n";
    script << "\n";
    
    for (const auto& command : plot_commands_) {
        script << command << "\n";
    }
    
    script << "\nplt.tight_layout()\n";
    script << "plt.show()\n";
}

void PythonExportBackend::export_vector(const Vector& data, const std::string& filename) {
    std::ofstream file(filename);
    for (int i = 0; i < data.size(); ++i) {
        file << data(i);
        if (i < data.size() - 1) file << "\n";
    }
}

void PythonExportBackend::export_matrix(const Matrix& data, const std::string& filename) {
    std::ofstream file(filename);
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            file << data(i, j);
            if (j < data.cols() - 1) file << ",";
        }
        if (i < data.rows() - 1) file << "\n";
    }
}

std::string PythonExportBackend::config_to_python_args(const PlotConfig& config, const std::string& label) {
    std::stringstream args;
    
    if (!label.empty()) {
        args << ", label='" << label << "'";
    }
    if (!config.color.empty() && config.color != "blue") {
        args << ", color='" << config.color << "'";
    }
    if (!config.linestyle.empty() && config.linestyle != "-") {
        args << ", linestyle='" << config.linestyle << "'";
    }
    if (!config.marker.empty()) {
        args << ", marker='" << config.marker << "'";
    }
    if (config.linewidth != 1.0f) {
        args << ", linewidth=" << config.linewidth;
    }
    if (config.markersize != 3.0f && !config.marker.empty()) {
        args << ", markersize=" << config.markersize;
    }
    
    return args.str();
}

// Plotter Implementation
Plotter::Plotter(std::unique_ptr<PlotBackend> backend) : backend_(std::move(backend)) {}

void Plotter::set_backend(std::unique_ptr<PlotBackend> backend) {
    backend_ = std::move(backend);
}

void Plotter::plot(const Vector& x, const Vector& y, const PlotConfig& config, const std::string& label) {
    backend_->plot_line(x, y, config, label);
}

void Plotter::scatter(const Vector& x, const Vector& y, const PlotConfig& config, const std::string& label) {
    backend_->plot_scatter(x, y, config, label);
}

void Plotter::heatmap(const Matrix& data, const PlotConfig& config) {
    backend_->plot_heatmap(data, config);
}

void Plotter::histogram(const Vector& data, int bins, const PlotConfig& config) {
    backend_->plot_histogram(data, bins, config);
}

void Plotter::save(const std::string& filename, int dpi) {
    backend_->save_plot(filename, dpi);
}

void Plotter::show() {
    backend_->show_plot();
}

void Plotter::clear() {
    backend_->clear_plot();
}

void Plotter::subplot(int rows, int cols, int index) {
    backend_->subplot(rows, cols, index);
}

void Plotter::plot_timeseries(const Matrix& data, const PlotConfig& config) {
    Vector time = Vector::LinSpaced(data.rows(), 0, data.rows() - 1);
    
    for (int i = 0; i < data.cols(); ++i) {
        Vector series = data.col(i);
        PlotConfig series_config = config;
        series_config.color = (i == 0) ? "blue" : (i == 1) ? "red" : "green";
        
        plot(time, series, series_config, "Series " + std::to_string(i + 1));
    }
}

void Plotter::plot_reservoir_states(const Matrix& states, const PlotConfig& config) {
    // Plot first few reservoir units over time
    int max_units = std::min(5, static_cast<int>(states.cols()));
    
    Vector time = Vector::LinSpaced(states.rows(), 0, states.rows() - 1);
    
    for (int i = 0; i < max_units; ++i) {
        Vector unit_activity = states.col(i);
        PlotConfig unit_config = config;
        
        plot(time, unit_activity, unit_config, "Unit " + std::to_string(i + 1));
    }
}

void Plotter::plot_weight_matrix(const Matrix& weights, const PlotConfig& config) {
    heatmap(weights, config);
}

void Plotter::plot_training_loss(const Vector& train_loss, const Vector& val_loss, const PlotConfig& config) {
    Vector epochs = Vector::LinSpaced(train_loss.size(), 1, train_loss.size());
    
    PlotConfig train_config = config;
    train_config.color = "blue";
    plot(epochs, train_loss, train_config, "Training Loss");
    
    if (val_loss.size() > 0) {
        PlotConfig val_config = config;
        val_config.color = "red";
        val_config.linestyle = "--";
        plot(epochs, val_loss, val_config, "Validation Loss");
    }
}

// PlotUtils Implementation
std::unique_ptr<Plotter> PlotUtils::default_plotter_ = nullptr;
std::string PlotUtils::default_backend_type_ = "python_export";
std::string PlotUtils::default_backend_args_ = "plots";

Plotter& PlotUtils::get_default_plotter() {
    if (!default_plotter_) {
        auto backend = create_backend(default_backend_type_, default_backend_args_);
        default_plotter_ = std::make_unique<Plotter>(std::move(backend));
    }
    return *default_plotter_;
}

void PlotUtils::set_default_backend(const std::string& backend_type, const std::string& args) {
    default_backend_type_ = backend_type;
    default_backend_args_ = args;
    
    // Reset default plotter to use new backend
    default_plotter_.reset();
}

std::unique_ptr<PlotBackend> PlotUtils::create_backend(const std::string& backend_type, const std::string& args) {
    if (backend_type == "python_export") {
        return std::make_unique<PythonExportBackend>(args.empty() ? "plots" : args);
    } else if (backend_type == "null") {
        return std::make_unique<NullBackend>();
    } else {
        // Default to Python export
        return std::make_unique<PythonExportBackend>(args.empty() ? "plots" : args);
    }
}

void PlotUtils::quick_plot(const Vector& y, const PlotConfig& config) {
    Vector x = Vector::LinSpaced(y.size(), 0, y.size() - 1);
    get_default_plotter().plot(x, y, config);
    get_default_plotter().show();
}

void PlotUtils::quick_scatter(const Vector& x, const Vector& y, const PlotConfig& config) {
    get_default_plotter().scatter(x, y, config);
    get_default_plotter().show();
}

} // namespace plotting
} // namespace reservoircpp