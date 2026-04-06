# Requirements Document: Vault-Synth Quantum Privacy Engine

## Introduction

Vault-Synth is a quantum privacy engine that generates synthetic financial data using a Quantum Circuit Born Machine (QCBM). The system takes sensitive customer transaction data and produces statistically identical synthetic data that can be used for AI training while maintaining privacy compliance with regulations like GDPR. The synthetic data preserves the statistical properties of the original dataset without containing any real customer information.

## Glossary

- **QCBM**: Quantum Circuit Born Machine - a quantum generative model that learns probability distributions
- **System**: The Vault-Synth quantum privacy engine
- **Data_Loader**: Component responsible for loading and preprocessing input data
- **Quantum_Circuit**: The quantum circuit implementation using PennyLane
- **Generator**: Component that produces synthetic samples from the trained circuit
- **Optimizer**: The Adam optimization algorithm used for training
- **MMD**: Maximum Mean Discrepancy - a statistical distance measure between distributions
- **Synthetic_Data**: Generated data that is statistically similar to real data but contains no real customer information
- **Real_Data**: Original sensitive customer transaction data
- **Discretization**: Process of converting continuous values into discrete bins
- **Normalization**: Process of scaling values to a specific range (0 to π)

## Requirements

### Requirement 1: Data Loading and Preprocessing

**User Story:** As a data scientist, I want to load and preprocess credit card transaction data, so that it can be fed into the quantum circuit for training.

#### Acceptance Criteria

1. WHEN a CSV file path is provided, THE Data_Loader SHALL load the file into memory
2. WHEN the data is loaded, THE Data_Loader SHALL select exactly 6 numerical columns from the dataset
3. WHEN numerical columns are selected, THE Data_Loader SHALL normalize all values to the range [0, π]
4. WHEN values are normalized, THE Data_Loader SHALL discretize the continuous values into discrete bins
5. IF the CSV file does not exist or is corrupted, THEN THE Data_Loader SHALL return a descriptive error message

### Requirement 2: Quantum Circuit Architecture

**User Story:** As a quantum engineer, I want to define a quantum circuit with appropriate architecture, so that it can learn the probability distribution of financial data.

#### Acceptance Criteria

1. THE Quantum_Circuit SHALL use between 4 and 12 qubits based on hardware constraints
2. THE Quantum_Circuit SHALL implement StronglyEntanglingLayers ansatz with exactly 3 layers
3. WHEN the circuit is executed, THE Quantum_Circuit SHALL output probability distributions using qml.probs()
4. THE Quantum_Circuit SHALL run on a PennyLane classical simulator device
5. WHEN circuit parameters are initialized, THE Quantum_Circuit SHALL use random initialization for weights

### Requirement 3: Training Process

**User Story:** As a machine learning engineer, I want to train the quantum circuit on real data, so that it learns to generate statistically similar synthetic data.

#### Acceptance Criteria

1. WHEN training begins, THE System SHALL use the Adam optimizer for parameter updates
2. WHEN calculating loss, THE System SHALL compute Maximum Mean Discrepancy (MMD) with a Gaussian kernel
3. THE System SHALL execute exactly 200 training steps
4. WHEN each training step completes, THE System SHALL update circuit parameters based on the computed gradient
5. WHEN training completes, THE System SHALL store the optimized circuit parameters

### Requirement 4: Synthetic Data Generation

**User Story:** As a data scientist, I want to generate synthetic transaction data, so that I can use it for AI training without privacy concerns.

#### Acceptance Criteria

1. WHEN the circuit is trained, THE Generator SHALL produce synthetic samples by sampling from the learned probability distribution
2. WHEN generating samples, THE Generator SHALL convert discrete bins back to continuous values in the original data range
3. WHEN a number of samples is requested, THE Generator SHALL produce exactly that number of synthetic samples
4. THE Generator SHALL produce synthetic data that maintains no direct correspondence to any real customer record
5. WHEN generating data, THE Generator SHALL operate independently of the input training data

### Requirement 5: Hardware Optimization

**User Story:** As a developer, I want the system to run efficiently on laptop hardware, so that it can be used without requiring specialized quantum computers or cloud resources.

#### Acceptance Criteria

1. THE System SHALL execute successfully on hardware with 16GB RAM
2. WHEN performing computations, THE System SHALL use vectorized operations for numerical calculations
3. THE System SHALL use a classical simulator rather than requiring quantum hardware
4. WHEN processing data, THE System SHALL manage memory efficiently to prevent out-of-memory errors
5. THE System SHALL complete training within a reasonable time frame on laptop hardware (under 30 minutes)

### Requirement 6: Statistical Similarity Validation

**User Story:** As a compliance officer, I want to verify that synthetic data is statistically similar to real data, so that I can ensure it is useful for downstream tasks.

#### Acceptance Criteria

1. WHEN comparing distributions, THE System SHALL compute MMD loss between real and synthetic data
2. WHEN training converges, THE System SHALL achieve an MMD loss below a defined threshold
3. THE System SHALL provide metrics comparing statistical properties of real and synthetic data
4. WHEN validation is performed, THE System SHALL verify that key statistical moments (mean, variance) are preserved

### Requirement 7: Privacy Guarantees

**User Story:** As a privacy officer, I want to ensure synthetic data contains no real customer information, so that we maintain GDPR compliance.

#### Acceptance Criteria

1. THE Generator SHALL produce data that cannot be traced back to individual customer records
2. WHEN synthetic data is generated, THE System SHALL ensure no direct copying of real data samples
3. THE System SHALL generate data from learned probability distributions rather than memorized samples
4. WHEN privacy is evaluated, THE System SHALL demonstrate that synthetic samples are novel and not present in the training set

### Requirement 8: Error Handling and Robustness

**User Story:** As a developer, I want the system to handle errors gracefully, so that users receive clear feedback when issues occur.

#### Acceptance Criteria

1. IF input data contains invalid values (NaN, infinity), THEN THE Data_Loader SHALL reject the data and return an error message
2. IF the number of qubits exceeds hardware capacity, THEN THE System SHALL return a descriptive error message
3. IF training fails to converge, THEN THE System SHALL log the issue and provide diagnostic information
4. WHEN any component encounters an error, THE System SHALL provide clear error messages indicating the failure point
5. IF insufficient data is provided for training, THEN THE System SHALL reject the training request with an appropriate error message

### Requirement 9: Configuration and Flexibility

**User Story:** As a researcher, I want to configure key parameters of the system, so that I can experiment with different settings.

#### Acceptance Criteria

1. THE System SHALL allow configuration of the number of qubits (between 4 and 12)
2. THE System SHALL allow configuration of the number of training steps
3. THE System SHALL allow configuration of the number of ansatz layers
4. THE System SHALL allow configuration of the learning rate for the optimizer
5. WHERE custom configurations are provided, THE System SHALL validate parameters before execution
