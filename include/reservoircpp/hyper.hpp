/**
 * @file hyper.hpp
 * @brief Hyperparameter optimization utilities for ReservoirCpp
 * 
 * This module provides utilities for hyperparameter optimization and tuning
 * of reservoir computing models.
 * 
 * @author ReservoirCpp Development Team
 * @date 2024
 * @license MIT License
 */

#ifndef RESERVOIRCPP_HYPER_HPP
#define RESERVOIRCPP_HYPER_HPP

#include "reservoircpp/types.hpp"
#include "reservoircpp/node.hpp"
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_map>
#include <random>

namespace reservoircpp {
namespace hyper {

/**
 * @brief Parameter search space definition
 */
struct ParameterSpace {
    enum class Type { UNIFORM, LOG_UNIFORM, CHOICE, NORMAL };
    
    Type type;
    std::string name;
    float min_val = 0.0f;
    float max_val = 1.0f;
    float mean = 0.0f;
    float std = 1.0f;
    std::vector<float> choices;
    
    ParameterSpace(const std::string& param_name, Type param_type)
        : name(param_name), type(param_type) {}
    
    // Factory methods
    static ParameterSpace uniform(const std::string& name, float min_val, float max_val);
    static ParameterSpace log_uniform(const std::string& name, float min_val, float max_val);
    static ParameterSpace choice(const std::string& name, const std::vector<float>& choices);
    static ParameterSpace normal(const std::string& name, float mean, float std);
};

/**
 * @brief Hyperparameter configuration
 */
using HyperConfig = std::unordered_map<std::string, float>;

/**
 * @brief Optimization result
 */
struct OptimizationResult {
    HyperConfig best_params;
    float best_score;
    std::vector<HyperConfig> all_params;
    std::vector<float> all_scores;
    int n_trials;
    float optimization_time;
    
    OptimizationResult() : best_score(-std::numeric_limits<float>::infinity()), 
                          n_trials(0), optimization_time(0.0f) {}
};

/**
 * @brief Objective function type
 * Takes hyperparameters and returns a score to maximize
 */
using ObjectiveFunction = std::function<float(const HyperConfig&)>;

/**
 * @brief Base class for hyperparameter optimization algorithms
 */
class BaseOptimizer {
public:
    /**
     * @brief Constructor
     * @param search_space Vector of parameter spaces to search
     * @param seed Random seed
     */
    BaseOptimizer(const std::vector<ParameterSpace>& search_space, 
                  unsigned int seed = 42);

    virtual ~BaseOptimizer() = default;

    /**
     * @brief Run optimization
     * @param objective Objective function to maximize
     * @param n_trials Number of trials to run
     * @return Optimization results
     */
    virtual OptimizationResult optimize(const ObjectiveFunction& objective, 
                                      int n_trials) = 0;

    /**
     * @brief Sample parameters from search space
     * @return Sampled hyperparameter configuration
     */
    virtual HyperConfig sample_params() = 0;

    // Getters
    const std::vector<ParameterSpace>& get_search_space() const { return search_space_; }
    unsigned int get_seed() const { return seed_; }

protected:
    std::vector<ParameterSpace> search_space_;
    unsigned int seed_;
    std::mt19937 rng_;
    
    /**
     * @brief Sample from a single parameter space
     */
    float sample_parameter(const ParameterSpace& space);
};

/**
 * @brief Random search optimizer
 */
class RandomSearch : public BaseOptimizer {
public:
    RandomSearch(const std::vector<ParameterSpace>& search_space, 
                 unsigned int seed = 42);

    OptimizationResult optimize(const ObjectiveFunction& objective, 
                              int n_trials) override;

    HyperConfig sample_params() override;
};

/**
 * @brief Grid search optimizer
 */
class GridSearch : public BaseOptimizer {
public:
    /**
     * @brief Constructor
     * @param search_space Parameter spaces (only uniform and choice supported)
     * @param n_points Number of points per dimension for uniform spaces
     */
    GridSearch(const std::vector<ParameterSpace>& search_space, 
               int n_points = 10);

    OptimizationResult optimize(const ObjectiveFunction& objective, 
                              int n_trials) override;

    HyperConfig sample_params() override;

private:
    int n_points_;
    std::vector<std::vector<float>> grid_points_;
    size_t current_index_;
    
    void generate_grid();
    std::vector<float> generate_uniform_points(const ParameterSpace& space);
};

/**
 * @brief Bayesian optimization optimizer (simplified version)
 */
class BayesianOptimization : public BaseOptimizer {
public:
    /**
     * @brief Constructor
     * @param search_space Parameter spaces
     * @param n_initial_points Number of random initial points
     * @param acquisition_function Acquisition function ("ei" or "ucb")
     */
    BayesianOptimization(const std::vector<ParameterSpace>& search_space, 
                        int n_initial_points = 10,
                        const std::string& acquisition_function = "ei");

    OptimizationResult optimize(const ObjectiveFunction& objective, 
                              int n_trials) override;

    HyperConfig sample_params() override;

private:
    int n_initial_points_;
    std::string acquisition_function_;
    std::vector<HyperConfig> observed_params_;
    std::vector<float> observed_scores_;
    
    /**
     * @brief Acquisition function for next point selection
     */
    float acquisition_score(const HyperConfig& params);
    
    /**
     * @brief Simple Gaussian Process prediction (stub)
     */
    std::pair<float, float> gp_predict(const HyperConfig& params);
};

/**
 * @brief Hyperparameter research utilities
 */
class HyperResearch {
public:
    /**
     * @brief Create a research study
     * @param study_name Name of the study
     * @param search_space Parameter search space
     * @param optimizer_type Type of optimizer ("random", "grid", "bayesian")
     * @return Optimizer instance
     */
    static std::unique_ptr<BaseOptimizer> create_study(
        const std::string& study_name,
        const std::vector<ParameterSpace>& search_space,
        const std::string& optimizer_type = "random");

    /**
     * @brief Run hyperparameter optimization for a model
     * @param model_factory Function that creates model from hyperparameters
     * @param evaluation_function Function that evaluates model performance
     * @param search_space Parameter search space
     * @param n_trials Number of optimization trials
     * @param optimizer_type Type of optimizer to use
     * @return Optimization results
     */
    static OptimizationResult optimize_model(
        std::function<std::unique_ptr<Node>(const HyperConfig&)> model_factory,
        std::function<float(const Node&)> evaluation_function,
        const std::vector<ParameterSpace>& search_space,
        int n_trials = 100,
        const std::string& optimizer_type = "random");

    /**
     * @brief Cross-validation evaluation
     * @param model Model to evaluate
     * @param X Input data
     * @param y Target data
     * @param n_folds Number of CV folds
     * @return Average CV score
     */
    static float cross_validate(const Node& model, 
                               const Matrix& X, 
                               const Matrix& y, 
                               int n_folds = 5);
};

/**
 * @brief Hyperparameter optimization report
 */
struct OptimizationReport {
    OptimizationResult result;
    std::string study_name;
    std::string optimizer_type;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    
    /**
     * @brief Save report to file
     * @param filename Output filename
     */
    void save(const std::string& filename) const;
    
    /**
     * @brief Load report from file
     * @param filename Input filename
     * @return Loaded report
     */
    static OptimizationReport load(const std::string& filename);
    
    /**
     * @brief Print summary to console
     */
    void print_summary() const;
};

/**
 * @brief Plotting utilities for optimization results
 */
class PlotUtils {
public:
    /**
     * @brief Plot optimization convergence
     * @param result Optimization results
     * @param filename Output filename (optional)
     */
    static void plot_convergence(const OptimizationResult& result, 
                                const std::string& filename = "");

    /**
     * @brief Plot parameter importance
     * @param result Optimization results
     * @param filename Output filename (optional)
     */
    static void plot_parameter_importance(const OptimizationResult& result,
                                        const std::string& filename = "");

    /**
     * @brief Plot parameter correlations
     * @param result Optimization results
     * @param filename Output filename (optional)
     */
    static void plot_param_correlations(const OptimizationResult& result,
                                      const std::string& filename = "");

    /**
     * @brief Export data for Python plotting
     * @param result Optimization results
     * @param directory Output directory
     */
    static void export_for_python_plotting(const OptimizationResult& result,
                                          const std::string& directory);
};

} // namespace hyper
} // namespace reservoircpp

#endif // RESERVOIRCPP_HYPER_HPP