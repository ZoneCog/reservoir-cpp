/**
 * @file hyper.cpp
 * @brief Implementation of hyperparameter optimization utilities
 */

#include "reservoircpp/hyper.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>

namespace reservoircpp {
namespace hyper {

// ParameterSpace Implementation
ParameterSpace ParameterSpace::uniform(const std::string& name, float min_val, float max_val) {
    ParameterSpace space(name, Type::UNIFORM);
    space.min_val = min_val;
    space.max_val = max_val;
    return space;
}

ParameterSpace ParameterSpace::log_uniform(const std::string& name, float min_val, float max_val) {
    ParameterSpace space(name, Type::LOG_UNIFORM);
    space.min_val = std::log(min_val);
    space.max_val = std::log(max_val);
    return space;
}

ParameterSpace ParameterSpace::choice(const std::string& name, const std::vector<float>& choices) {
    ParameterSpace space(name, Type::CHOICE);
    space.choices = choices;
    return space;
}

ParameterSpace ParameterSpace::normal(const std::string& name, float mean, float std) {
    ParameterSpace space(name, Type::NORMAL);
    space.mean = mean;
    space.std = std;
    return space;
}

// BaseOptimizer Implementation
BaseOptimizer::BaseOptimizer(const std::vector<ParameterSpace>& search_space, unsigned int seed)
    : search_space_(search_space), seed_(seed), rng_(seed) {
}

float BaseOptimizer::sample_parameter(const ParameterSpace& space) {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(space.mean, space.std);
    
    switch (space.type) {
        case ParameterSpace::Type::UNIFORM:
            return space.min_val + uniform(rng_) * (space.max_val - space.min_val);
            
        case ParameterSpace::Type::LOG_UNIFORM: {
            float log_val = space.min_val + uniform(rng_) * (space.max_val - space.min_val);
            return std::exp(log_val);
        }
        
        case ParameterSpace::Type::CHOICE: {
            if (space.choices.empty()) return 0.0f;
            std::uniform_int_distribution<size_t> choice_dist(0, space.choices.size() - 1);
            return space.choices[choice_dist(rng_)];
        }
        
        case ParameterSpace::Type::NORMAL:
            return normal(rng_);
            
        default:
            return 0.0f;
    }
}

// RandomSearch Implementation
RandomSearch::RandomSearch(const std::vector<ParameterSpace>& search_space, unsigned int seed)
    : BaseOptimizer(search_space, seed) {
}

OptimizationResult RandomSearch::optimize(const ObjectiveFunction& objective, int n_trials) {
    OptimizationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int trial = 0; trial < n_trials; ++trial) {
        HyperConfig params = sample_params();
        float score = objective(params);
        
        result.all_params.push_back(params);
        result.all_scores.push_back(score);
        
        if (score > result.best_score) {
            result.best_score = score;
            result.best_params = params;
        }
        
        // Progress reporting
        if ((trial + 1) % std::max(1, n_trials / 10) == 0) {
            std::cout << "Trial " << (trial + 1) << "/" << n_trials 
                      << ", Best score: " << result.best_score << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    result.n_trials = n_trials;
    result.optimization_time = duration.count() / 1000.0f;
    
    return result;
}

HyperConfig RandomSearch::sample_params() {
    HyperConfig config;
    
    for (const auto& space : search_space_) {
        config[space.name] = sample_parameter(space);
    }
    
    return config;
}

// GridSearch Implementation
GridSearch::GridSearch(const std::vector<ParameterSpace>& search_space, int n_points)
    : BaseOptimizer(search_space, 0), n_points_(n_points), current_index_(0) {
    generate_grid();
}

void GridSearch::generate_grid() {
    grid_points_.clear();
    
    for (const auto& space : search_space_) {
        if (space.type == ParameterSpace::Type::CHOICE) {
            grid_points_.push_back(space.choices);
        } else if (space.type == ParameterSpace::Type::UNIFORM) {
            grid_points_.push_back(generate_uniform_points(space));
        } else {
            // For other types, fall back to uniform sampling
            ParameterSpace uniform_space = ParameterSpace::uniform(space.name, 
                                                                  space.min_val, 
                                                                  space.max_val);
            grid_points_.push_back(generate_uniform_points(uniform_space));
        }
    }
}

std::vector<float> GridSearch::generate_uniform_points(const ParameterSpace& space) {
    std::vector<float> points;
    
    for (int i = 0; i < n_points_; ++i) {
        float t = static_cast<float>(i) / (n_points_ - 1);
        float value = space.min_val + t * (space.max_val - space.min_val);
        points.push_back(value);
    }
    
    return points;
}

OptimizationResult GridSearch::optimize(const ObjectiveFunction& objective, int n_trials) {
    OptimizationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Calculate total grid points
    size_t total_points = 1;
    for (const auto& points : grid_points_) {
        total_points *= points.size();
    }
    
    size_t max_trials = std::min(static_cast<size_t>(n_trials), total_points);
    
    for (size_t trial = 0; trial < max_trials; ++trial) {
        HyperConfig params = sample_params();
        float score = objective(params);
        
        result.all_params.push_back(params);
        result.all_scores.push_back(score);
        
        if (score > result.best_score) {
            result.best_score = score;
            result.best_params = params;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    result.n_trials = max_trials;
    result.optimization_time = duration.count() / 1000.0f;
    
    return result;
}

HyperConfig GridSearch::sample_params() {
    HyperConfig config;
    
    if (grid_points_.empty()) {
        return config;
    }
    
    // Convert current_index to multi-dimensional grid coordinates
    size_t temp_index = current_index_;
    
    for (size_t i = 0; i < search_space_.size(); ++i) {
        size_t grid_size = grid_points_[i].size();
        size_t coord = temp_index % grid_size;
        temp_index /= grid_size;
        
        config[search_space_[i].name] = grid_points_[i][coord];
    }
    
    current_index_++;
    return config;
}

// BayesianOptimization Implementation (simplified)
BayesianOptimization::BayesianOptimization(const std::vector<ParameterSpace>& search_space, 
                                         int n_initial_points,
                                         const std::string& acquisition_function)
    : BaseOptimizer(search_space, 42), n_initial_points_(n_initial_points),
      acquisition_function_(acquisition_function) {
}

OptimizationResult BayesianOptimization::optimize(const ObjectiveFunction& objective, int n_trials) {
    OptimizationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initial random exploration
    RandomSearch random_search(search_space_, seed_);
    for (int i = 0; i < n_initial_points_; ++i) {
        HyperConfig params = random_search.sample_params();
        float score = objective(params);
        
        observed_params_.push_back(params);
        observed_scores_.push_back(score);
        
        result.all_params.push_back(params);
        result.all_scores.push_back(score);
        
        if (score > result.best_score) {
            result.best_score = score;
            result.best_params = params;
        }
    }
    
    // Bayesian optimization loop (simplified)
    for (int trial = n_initial_points_; trial < n_trials; ++trial) {
        // For simplicity, continue with random search
        // Real implementation would use Gaussian Process and acquisition function
        HyperConfig params = random_search.sample_params();
        float score = objective(params);
        
        observed_params_.push_back(params);
        observed_scores_.push_back(score);
        
        result.all_params.push_back(params);
        result.all_scores.push_back(score);
        
        if (score > result.best_score) {
            result.best_score = score;
            result.best_params = params;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    result.n_trials = n_trials;
    result.optimization_time = duration.count() / 1000.0f;
    
    return result;
}

HyperConfig BayesianOptimization::sample_params() {
    RandomSearch random_search(search_space_, seed_);
    return random_search.sample_params();
}

float BayesianOptimization::acquisition_score(const HyperConfig& params) {
    // Simplified acquisition function
    auto [mean, variance] = gp_predict(params);
    
    if (acquisition_function_ == "ei") {
        // Expected Improvement (simplified)
        float best_score = *std::max_element(observed_scores_.begin(), observed_scores_.end());
        float improvement = mean - best_score;
        return improvement > 0 ? improvement * variance : 0.0f;
    } else {
        // Upper Confidence Bound
        return mean + 2.0f * std::sqrt(variance);
    }
}

std::pair<float, float> BayesianOptimization::gp_predict(const HyperConfig& params) {
    // Simplified GP prediction - would need proper implementation
    float mean = 0.0f;
    float variance = 1.0f;
    
    if (!observed_scores_.empty()) {
        mean = *std::max_element(observed_scores_.begin(), observed_scores_.end());
        variance = 0.1f;
    }
    
    return {mean, variance};
}

// HyperResearch Implementation
std::unique_ptr<BaseOptimizer> HyperResearch::create_study(
    const std::string& study_name,
    const std::vector<ParameterSpace>& search_space,
    const std::string& optimizer_type) {
    
    if (optimizer_type == "random") {
        return std::make_unique<RandomSearch>(search_space);
    } else if (optimizer_type == "grid") {
        return std::make_unique<GridSearch>(search_space);
    } else if (optimizer_type == "bayesian") {
        return std::make_unique<BayesianOptimization>(search_space);
    } else {
        return std::make_unique<RandomSearch>(search_space);  // Default
    }
}

OptimizationResult HyperResearch::optimize_model(
    std::function<std::unique_ptr<Node>(const HyperConfig&)> model_factory,
    std::function<float(const Node&)> evaluation_function,
    const std::vector<ParameterSpace>& search_space,
    int n_trials,
    const std::string& optimizer_type) {
    
    auto optimizer = create_study("model_optimization", search_space, optimizer_type);
    
    ObjectiveFunction objective = [&](const HyperConfig& params) -> float {
        auto model = model_factory(params);
        return evaluation_function(*model);
    };
    
    return optimizer->optimize(objective, n_trials);
}

float HyperResearch::cross_validate(const Node& model, const Matrix& X, const Matrix& y, int n_folds) {
    // Simplified cross-validation implementation
    if (X.rows() != y.rows()) {
        throw std::runtime_error("X and y must have same number of rows");
    }
    
    size_t n_samples = X.rows();
    size_t fold_size = n_samples / n_folds;
    
    std::vector<float> scores;
    
    for (int fold = 0; fold < n_folds; ++fold) {
        size_t start_idx = fold * fold_size;
        size_t end_idx = (fold == n_folds - 1) ? n_samples : (fold + 1) * fold_size;
        
        // Simple evaluation (would need proper train/test split and evaluation)
        float score = 0.5f + 0.1f * fold;  // Placeholder
        scores.push_back(score);
    }
    
    // Return average score
    float sum = 0.0f;
    for (float score : scores) {
        sum += score;
    }
    
    return sum / scores.size();
}

// OptimizationReport Implementation
void OptimizationReport::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }
    
    file << "Optimization Report\n";
    file << "==================\n";
    file << "Study: " << study_name << "\n";
    file << "Optimizer: " << optimizer_type << "\n";
    file << "Trials: " << result.n_trials << "\n";
    file << "Best Score: " << result.best_score << "\n";
    file << "Optimization Time: " << result.optimization_time << " seconds\n";
    
    file << "\nBest Parameters:\n";
    for (const auto& [key, value] : result.best_params) {
        file << "  " << key << ": " << value << "\n";
    }
}

OptimizationReport OptimizationReport::load(const std::string& filename) {
    OptimizationReport report;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        return report;
    }
    
    // Simple parsing - would need proper implementation
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("Study: ") == 0) {
            report.study_name = line.substr(7);
        } else if (line.find("Best Score: ") == 0) {
            report.result.best_score = std::stof(line.substr(12));
        }
    }
    
    return report;
}

void OptimizationReport::print_summary() const {
    std::cout << "Optimization Report\n";
    std::cout << "==================\n";
    std::cout << "Study: " << study_name << "\n";
    std::cout << "Optimizer: " << optimizer_type << "\n";
    std::cout << "Trials: " << result.n_trials << "\n";
    std::cout << "Best Score: " << result.best_score << "\n";
    std::cout << "Optimization Time: " << result.optimization_time << " seconds\n";
    
    std::cout << "\nBest Parameters:\n";
    for (const auto& [key, value] : result.best_params) {
        std::cout << "  " << key << ": " << value << "\n";
    }
}

// PlotUtils Implementation
void PlotUtils::plot_convergence(const OptimizationResult& result, const std::string& filename) {
    // Export data for external plotting
    if (!filename.empty()) {
        std::ofstream file(filename + "_convergence.csv");
        file << "trial,score,best_score\n";
        
        float best_so_far = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < result.all_scores.size(); ++i) {
            best_so_far = std::max(best_so_far, result.all_scores[i]);
            file << i << "," << result.all_scores[i] << "," << best_so_far << "\n";
        }
    }
}

void PlotUtils::plot_parameter_importance(const OptimizationResult& result, const std::string& filename) {
    // Export parameter data for analysis
    if (!filename.empty() && !result.all_params.empty()) {
        std::ofstream file(filename + "_params.csv");
        
        // Header
        file << "trial,score";
        for (const auto& [key, value] : result.all_params[0]) {
            file << "," << key;
        }
        file << "\n";
        
        // Data
        for (size_t i = 0; i < result.all_params.size(); ++i) {
            file << i << "," << result.all_scores[i];
            for (const auto& [key, value] : result.all_params[i]) {
                file << "," << value;
            }
            file << "\n";
        }
    }
}

void PlotUtils::plot_param_correlations(const OptimizationResult& result, const std::string& filename) {
    // Placeholder for correlation analysis
    std::cout << "Parameter correlation analysis would be implemented here\n";
}

void PlotUtils::export_for_python_plotting(const OptimizationResult& result, const std::string& directory) {
    // Create directory if it doesn't exist (manual implementation for compatibility)
    std::string mkdir_cmd = "mkdir -p " + directory;
    system(mkdir_cmd.c_str());
    
    plot_convergence(result, directory + "/optimization");
    plot_parameter_importance(result, directory + "/optimization");
    
    // Create Python plotting script
    std::ofstream script(directory + "/plot.py");
    script << "import pandas as pd\n";
    script << "import matplotlib.pyplot as plt\n";
    script << "\n";
    script << "# Load convergence data\n";
    script << "conv_data = pd.read_csv('optimization_convergence.csv')\n";
    script << "plt.figure(figsize=(10, 6))\n";
    script << "plt.subplot(1, 2, 1)\n";
    script << "plt.plot(conv_data['trial'], conv_data['best_score'])\n";
    script << "plt.title('Optimization Convergence')\n";
    script << "plt.xlabel('Trial')\n";
    script << "plt.ylabel('Best Score')\n";
    script << "\n";
    script << "# Load parameter data\n";
    script << "param_data = pd.read_csv('optimization_params.csv')\n";
    script << "plt.subplot(1, 2, 2)\n";
    script << "plt.scatter(param_data.iloc[:, 2], param_data['score'])\n";
    script << "plt.title('Parameter vs Score')\n";
    script << "plt.xlabel('Parameter Value')\n";
    script << "plt.ylabel('Score')\n";
    script << "\n";
    script << "plt.tight_layout()\n";
    script << "plt.savefig('optimization_plots.png')\n";
    script << "plt.show()\n";
}

} // namespace hyper
} // namespace reservoircpp