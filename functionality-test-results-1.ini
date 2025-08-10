cd /workspaces/reservoir-cpp && python3 detailed_verification.py
=== Detailed Python to C++ Functionality Verification ===
Sun Aug 10 18:19:51 UTC 2025
Date: 0

## Core Module Analysis
==================================================

### Analyzing activationsfunc.py
Found 8 functions and 0 classes
  Function 'vect_wrapper': ❌ NOT FOUND
  Function 'get_function': ✅ Found in reservoircpp/activations.hpp
  Function 'softmax': ✅ Found in reservoircpp/activations.hpp
  Function 'softplus': ✅ Found in reservoircpp/activations.hpp
  Function 'sigmoid': ✅ Found in reservoircpp/reservoir.hpp
  Function 'tanh': ✅ Found in reservoircpp/reservoir.hpp
  Function 'identity': ✅ Found in reservoircpp/node.hpp
  Function 'relu': ✅ Found in reservoircpp/activations.hpp

### Analyzing mat_gen.py
Found 2 functions and 1 classes
  Function 'rvs': ❌ NOT FOUND
  Function 'data_rvs': ❌ NOT FOUND
  Class 'Initializer': ❌ NOT FOUND

### Analyzing node.py
Found 42 functions and 2 classes
  Function 'input_dim': ✅ Found in reservoircpp/reservoir.hpp
  Function 'output_dim': ✅ Found in reservoircpp/gpu.hpp
  Function 'feedback_dim': ❌ NOT FOUND
  Function 'is_initialized': ✅ Found in reservoircpp/node.hpp
  Function 'has_feedback': ✅ Found in reservoircpp/node.hpp
  Function 'is_trained_offline': ❌ NOT FOUND
  Function 'is_trained_online': ❌ NOT FOUND
  Function 'is_trainable': ✅ Found in reservoircpp/node.hpp
  Function 'is_trainable': ✅ Found in reservoircpp/node.hpp
  Function 'fitted': ✅ Found in reservoircpp/reservoir.hpp
  Function 'is_fb_initialized': ❌ NOT FOUND
  Function 'dtype': ✅ Found in reservoircpp/node.hpp
  Function 'unsupervised': ❌ NOT FOUND
  Function 'state': ✅ Found in reservoircpp/reservoir.hpp
  Function 'state_proxy': ❌ NOT FOUND
  Function 'feedback': ✅ Found in reservoircpp/ops.hpp
  Function 'set_state_proxy': ❌ NOT FOUND
  Function 'set_input_dim': ✅ Found in reservoircpp/node.hpp
  Function 'set_output_dim': ✅ Found in reservoircpp/node.hpp
  Function 'set_feedback_dim': ❌ NOT FOUND
  Function 'get_param': ✅ Found in reservoircpp/node.hpp
  Function 'set_param': ✅ Found in reservoircpp/node.hpp
  Function 'create_buffer': ❌ NOT FOUND
  Function 'set_buffer': ❌ NOT FOUND
  Function 'get_buffer': ❌ NOT FOUND
  Function 'initialize': ✅ Found in reservoircpp/reservoir.hpp
  Function 'initialize_feedback': ❌ NOT FOUND
  Function 'initialize_buffers': ❌ NOT FOUND
  Function 'clean_buffers': ❌ NOT FOUND
  Function 'reset': ✅ Found in reservoircpp/reservoir.hpp
  Function 'with_state': ❌ NOT FOUND
  Function 'with_feedback': ✅ Found in reservoircpp/ops.hpp
  Function 'zero_state': ✅ Found in reservoircpp/node.hpp
  Function 'zero_feedback': ❌ NOT FOUND
  Function 'link_feedback': ✅ Found in reservoircpp/ops.hpp
  Function 'call': ✅ Found in reservoircpp/gpu.hpp
  Function 'run': ✅ Found in reservoircpp/model.hpp
  Function 'train': ✅ Found in reservoircpp/reservoir.hpp
  Function 'partial_fit': ✅ Found in reservoircpp/reservoir.hpp
  Function 'fit': ✅ Found in reservoircpp/reservoir.hpp
  Function 'copy': ✅ Found in reservoircpp/reservoir.hpp
  Function 'unsupervised': ❌ NOT FOUND
  Class 'Node': ✅ Found in reservoircpp/reservoir.hpp
  Class 'Unsupervised': ❌ NOT FOUND

### Analyzing model.py
Found 34 functions and 2 classes
  Function 'run_and_partial_fit': ❌ NOT FOUND
  Function 'run_submodel': ❌ NOT FOUND
  Function 'forward': ✅ Found in reservoircpp/reservoir.hpp
  Function 'train': ✅ Found in reservoircpp/reservoir.hpp
  Function 'initializer': ❌ NOT FOUND
  Function 'update_graph': ✅ Found in reservoircpp/model.hpp
  Function 'get_node': ✅ Found in reservoircpp/model.hpp
  Function 'nodes': ✅ Found in reservoircpp/observables.hpp
  Function 'node_names': ✅ Found in reservoircpp/model.hpp
  Function 'edges': ✅ Found in reservoircpp/model.hpp
  Function 'input_dim': ✅ Found in reservoircpp/reservoir.hpp
  Function 'output_dim': ✅ Found in reservoircpp/gpu.hpp
  Function 'input_nodes': ✅ Found in reservoircpp/model.hpp
  Function 'output_nodes': ✅ Found in reservoircpp/model.hpp
  Function 'trainable_nodes': ✅ Found in reservoircpp/model.hpp
  Function 'feedback_nodes': ✅ Found in reservoircpp/ops.hpp
  Function 'data_dispatcher': ✅ Found in reservoircpp/model.hpp
  Function 'is_empty': ✅ Found in reservoircpp/model.hpp
  Function 'is_trainable': ✅ Found in reservoircpp/node.hpp
  Function 'is_trainable': ✅ Found in reservoircpp/node.hpp
  Function 'is_trained_online': ❌ NOT FOUND
  Function 'is_trained_offline': ❌ NOT FOUND
  Function 'fitted': ✅ Found in reservoircpp/reservoir.hpp
  Function 'with_state': ❌ NOT FOUND
  Function 'with_feedback': ✅ Found in reservoircpp/ops.hpp
  Function 'reset': ✅ Found in reservoircpp/reservoir.hpp
  Function 'initialize': ✅ Found in reservoircpp/reservoir.hpp
  Function 'initialize_buffers': ❌ NOT FOUND
  Function 'call': ✅ Found in reservoircpp/gpu.hpp
  Function 'run': ✅ Found in reservoircpp/model.hpp
  Function 'train': ✅ Found in reservoircpp/reservoir.hpp
  Function 'fit': ✅ Found in reservoircpp/reservoir.hpp
  Function 'copy': ✅ Found in reservoircpp/reservoir.hpp
  Function 'update_graph': ✅ Found in reservoircpp/model.hpp
  Class 'Model': ✅ Found in reservoircpp/reservoir.hpp
  Class 'FrozenModel': ❌ NOT FOUND

### Analyzing ops.py
Found 4 functions and 0 classes
  Function 'concat_multi_inputs': ❌ NOT FOUND
  Function 'link': ✅ Found in reservoircpp/ops.hpp
  Function 'link_feedback': ✅ Found in reservoircpp/ops.hpp
  Function 'merge': ✅ Found in reservoircpp/model.hpp

### Analyzing observables.py
Found 7 functions and 0 classes
  Function 'spectral_radius': ✅ Found in reservoircpp/reservoir.hpp
  Function 'mse': ✅ Found in reservoircpp/observables.hpp
  Function 'rmse': ✅ Found in reservoircpp/observables.hpp
  Function 'nrmse': ✅ Found in reservoircpp/observables.hpp
  Function 'rsquare': ✅ Found in reservoircpp/observables.hpp
  Function 'memory_capacity': ✅ Found in reservoircpp/observables.hpp
  Function 'effective_spectral_radius': ✅ Found in reservoircpp/observables.hpp

### Analyzing type.py
Found 5 functions and 1 classes
  Function 'get_param': ✅ Found in reservoircpp/node.hpp
  Function 'initialize': ✅ Found in reservoircpp/reservoir.hpp
  Function 'reset': ✅ Found in reservoircpp/reservoir.hpp
  Function 'with_state': ❌ NOT FOUND
  Function 'with_feedback': ✅ Found in reservoircpp/ops.hpp
  Class 'NodeType': ✅ Found in reservoircpp/types.hpp


## Node Types Analysis
==================================================

### Analyzing nodes/delay.py
Found 2 functions and 1 classes
  Class 'Delay': ✅ Found in reservoircpp/reservoir.hpp

### Analyzing nodes/concat.py
Found 2 functions and 1 classes
  Class 'Concat': ✅ Found in reservoircpp/ops.hpp

### Analyzing nodes/esn.py
Found 7 functions and 1 classes
  Class 'ESN': ✅ Found in reservoircpp/reservoir.hpp

### Analyzing nodes/activations.py
Found 2 functions and 6 classes
  Class 'Softmax': ✅ Found in reservoircpp/activations.hpp
  Class 'Softplus': ✅ Found in reservoircpp/activations.hpp
  Class 'Sigmoid': ✅ Found in reservoircpp/reservoir.hpp
  Class 'Tanh': ✅ Found in reservoircpp/reservoir.hpp
  Class 'Identity': ✅ Found in reservoircpp/node.hpp
  Class 'ReLU': ✅ Found in reservoircpp/activations.hpp

### Analyzing nodes/io.py
Found 0 functions and 2 classes
  Class 'Input': ✅ Found in reservoircpp/reservoir.hpp
  Class 'Output': ✅ Found in reservoircpp/reservoir.hpp

### Analyzing nodes/reservoirs/reservoir.py
Found 0 functions and 1 classes
  Class 'Reservoir': ✅ Found in reservoircpp/reservoir.hpp

### Analyzing nodes/reservoirs/base.py
Found 5 functions and 0 classes

### Analyzing nodes/reservoirs/nvar.py
Found 2 functions and 1 classes
  Class 'NVAR': ✅ Found in reservoircpp/reservoir.hpp

### Analyzing nodes/reservoirs/intrinsic_plasticity.py
Found 9 functions and 1 classes
  Class 'IPReservoir': ❌ NOT FOUND

### Analyzing nodes/readouts/base.py
Found 1 functions and 0 classes


## Summary
==================================================
Functions: 73/102 implemented
Classes: 3/6 implemented
Overall: 76/108

⚠️  Some significant functionality may be missing.
❌ Manual verification recommended before migration.
