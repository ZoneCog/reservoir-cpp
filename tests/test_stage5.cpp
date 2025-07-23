/**
 * @file test_stage5.cpp
 * @brief Tests for Stage 5 - Ancillary and Advanced Features
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "reservoircpp/experimental.hpp"
#include "reservoircpp/compat.hpp"
#include "reservoircpp/hyper.hpp"
#include "reservoircpp/plotting.hpp"

using namespace reservoircpp;
using namespace Catch;

TEST_CASE("Stage 5 - Experimental - LIF neuron", "[stage5][experimental][lif]") {
    using namespace reservoircpp::experimental;
    
    SECTION("Basic construction") {
        LIF lif("test_lif", 10);
        
        REQUIRE(lif.name() == "test_lif");
        REQUIRE(lif.output_dim().size() == 1);
        REQUIRE(lif.output_dim()[0] == 10);
        REQUIRE(lif.get_tau_m() == 10.0f);
        REQUIRE(lif.get_threshold() == 1.0f);
    }
    
    SECTION("Forward pass") {
        LIF lif("test_lif", 5);
        Matrix input = Matrix::Constant(1, 5, 0.5);
        
        Matrix output = lif.forward(input);
        
        REQUIRE(output.rows() == 1);
        REQUIRE(output.cols() == 5);
        // Since input is below threshold, expect no spikes initially
        REQUIRE(output.sum() == 0.0);
    }
    
    SECTION("Spike generation") {
        LIF lif("test_lif", 3, 1.0f, 1.0f, 0.5f);  // Low threshold
        Matrix strong_input = Matrix::Constant(1, 3, 2.0);  // Strong input
        
        // Run multiple times to build up potential
        Matrix output;
        for (int i = 0; i < 10; ++i) {
            output = lif.forward(strong_input);
        }
        
        // Should eventually produce spikes
        REQUIRE(output.maxCoeff() == 1.0);
    }
}

TEST_CASE("Stage 5 - Experimental - Add node", "[stage5][experimental][add]") {
    using namespace reservoircpp::experimental;
    
    SECTION("Basic addition") {
        Add add_node("test_add");
        
        Matrix input1(2, 3);
        input1 << 1, 2, 3,
                  4, 5, 6;
        
        Matrix input2(2, 3);
        input2 << 1, 1, 1,
                  1, 1, 1;
        
        Matrix result = add_node.forward(input1, input2);
        
        REQUIRE(result.rows() == 2);
        REQUIRE(result.cols() == 3);
        REQUIRE(result(0, 0) == 2);
        REQUIRE(result(1, 2) == 7);
    }
    
    SECTION("Add with stored second input") {
        Add add_node("test_add");
        
        Matrix input1(1, 2);
        input1 << 10, 20;
        
        Matrix input2(1, 2);
        input2 << 5, 5;
        
        add_node.set_second_input(input2);
        Matrix result = add_node.forward(input1);
        
        REQUIRE(result(0, 0) == 15);
        REQUIRE(result(0, 1) == 25);
    }
}

TEST_CASE("Stage 5 - Experimental - RandomChoice", "[stage5][experimental][random]") {
    using namespace reservoircpp::experimental;
    
    SECTION("Feature selection") {
        RandomChoice choice("test_choice", 3, 42);
        
        Matrix input(2, 10);
        for (int i = 0; i < 10; ++i) {
            input(0, i) = i;
            input(1, i) = i * 10;
        }
        
        Matrix output = choice.forward(input);
        
        REQUIRE(output.rows() == 2);
        REQUIRE(output.cols() == 3);  // Selected 3 features
        
        // Run again with same input - should get same selection due to seed
        Matrix output2 = choice.forward(input);
        REQUIRE((output - output2).norm() < 1e-10);
    }
}

TEST_CASE("Stage 5 - Compatibility - Model serialization", "[stage5][compat]") {
    using namespace reservoircpp::compat;
    
    SECTION("Save and load config") {
        ModelConfig config;
        config.version = "0.4.0";
        config.model_type = "TestModel";
        config.parameters["param1"] = 1.5f;
        config.parameters["param2"] = 2.0f;
        
        std::string filename = "/tmp/test_config.json";
        REQUIRE(ModelSerializer::save_config(config, filename));
        
        ModelConfig loaded = ModelSerializer::load_config(filename);
        REQUIRE(loaded.version == "0.4.0");
        REQUIRE(loaded.model_type == "TestModel");
    }
    
    SECTION("Version compatibility") {
        REQUIRE(VersionInfo::is_supported("0.4.0"));
        REQUIRE(VersionInfo::is_supported("0.3.0"));
        REQUIRE(VersionInfo::is_supported("0.2.0"));
        REQUIRE_FALSE(VersionInfo::is_supported("0.1.0"));
        
        REQUIRE(VersionInfo::compare_versions("0.4.0", "0.3.0") > 0);
        REQUIRE(VersionInfo::compare_versions("0.2.0", "0.3.0") < 0);
        REQUIRE(VersionInfo::compare_versions("0.3.0", "0.3.0") == 0);
    }
}

TEST_CASE("Stage 5 - Hyperparameter optimization", "[stage5][hyper]") {
    using namespace reservoircpp::hyper;
    
    SECTION("Parameter space creation") {
        auto uniform_space = ParameterSpace::uniform("lr", 0.01f, 1.0f);
        REQUIRE(uniform_space.name == "lr");
        REQUIRE(uniform_space.type == ParameterSpace::Type::UNIFORM);
        REQUIRE(uniform_space.min_val == 0.01f);
        REQUIRE(uniform_space.max_val == 1.0f);
        
        auto choice_space = ParameterSpace::choice("units", {50, 100, 200});
        REQUIRE(choice_space.name == "units");
        REQUIRE(choice_space.type == ParameterSpace::Type::CHOICE);
        REQUIRE(choice_space.choices.size() == 3);
    }
    
    SECTION("Random search optimizer") {
        std::vector<ParameterSpace> search_space = {
            ParameterSpace::uniform("x", -1.0f, 1.0f),
            ParameterSpace::uniform("y", -1.0f, 1.0f)
        };
        
        RandomSearch optimizer(search_space, 42);
        
        // Simple quadratic objective: minimize x^2 + y^2
        ObjectiveFunction objective = [](const HyperConfig& params) -> float {
            float x = params.at("x");
            float y = params.at("y");
            return -(x*x + y*y);  // Negative because we maximize
        };
        
        OptimizationResult result = optimizer.optimize(objective, 10);
        
        REQUIRE(result.n_trials == 10);
        REQUIRE(result.all_params.size() == 10);
        REQUIRE(result.all_scores.size() == 10);
        REQUIRE(result.best_score <= 0.0f);  // Should be negative
        
        // Best point should be closer to origin
        float best_x = result.best_params.at("x");
        float best_y = result.best_params.at("y");
        REQUIRE(std::abs(best_x) <= 1.0f);
        REQUIRE(std::abs(best_y) <= 1.0f);
    }
}

TEST_CASE("Stage 5 - Plotting utilities", "[stage5][plotting]") {
    using namespace reservoircpp::plotting;
    
    SECTION("Plot configuration") {
        PlotConfig config;
        config.title = "Test Plot";
        config.xlabel = "X axis";
        config.ylabel = "Y axis";
        config.color = "red";
        
        REQUIRE(config.title == "Test Plot");
        REQUIRE(config.color == "red");
        REQUIRE(config.grid == true);
    }
    
    SECTION("Python export backend") {
        PythonExportBackend backend("/tmp/test_plots");
        
        Vector x = Vector::LinSpaced(5, 0, 4);
        Vector y = x.array().square();
        
        PlotConfig config;
        config.title = "Square function";
        
        // These should not throw
        REQUIRE_NOTHROW(backend.plot_line(x, y, config, "data"));
        REQUIRE_NOTHROW(backend.save_plot("test.png"));
        REQUIRE_NOTHROW(backend.generate_python_script());
    }
    
    SECTION("Plot utilities") {
        auto& plotter = PlotUtils::get_default_plotter();
        
        Vector data = Vector::Random(10);
        
        // Should not throw
        REQUIRE_NOTHROW(PlotUtils::quick_plot(data));
    }
}

TEST_CASE("Stage 5 - Integration test", "[stage5][integration]") {
    SECTION("All modules can be included together") {
        using namespace reservoircpp::experimental;
        using namespace reservoircpp::compat;
        using namespace reservoircpp::hyper;
        using namespace reservoircpp::plotting;
        
        // Create some experimental nodes
        LIF lif("integration_lif", 5);
        Add add("integration_add");
        RandomChoice choice("integration_choice", 3);
        
        // Create hyperparameter spaces
        std::vector<ParameterSpace> spaces = {
            ParameterSpace::uniform("param", 0.0f, 1.0f)
        };
        
        // Create optimization result
        OptimizationResult result;
        result.best_params["param"] = 0.5f;
        result.best_score = 0.8f;
        result.n_trials = 5;
        
        // Create plot configuration
        PlotConfig plot_config;
        plot_config.title = "Integration Test";
        
        // All should work together without conflicts
        REQUIRE(lif.name() == "integration_lif");
        REQUIRE(add.name() == "integration_add");
        REQUIRE(choice.name() == "integration_choice");
        REQUIRE(spaces.size() == 1);
        REQUIRE(result.best_score == 0.8f);
        REQUIRE(plot_config.title == "Integration Test");
    }
}