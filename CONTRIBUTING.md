# Contributing to Antibiotic Learning

Thank you for your interest in contributing to the Antibiotic Learning project!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Install dependencies: `pip install -r requirements.txt`
4. Install dev dependencies: `pip install pytest black flake8 isort`

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Follow the existing code style
- Add tests for new features
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 4. Format Code

```bash
# Format with black
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Check with flake8
flake8 src/ tests/ examples/
```

### 5. Commit Changes

```bash
git add .
git commit -m "Brief description of changes"
```

Follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style Guidelines

### Python Code

- Follow PEP 8
- Use type hints where helpful
- Write docstrings for all public functions/classes
- Keep functions focused and small
- Use descriptive variable names

### Docstring Format

```python
def function_name(param1, param2):
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        param1 (type): Description
        param2 (type): Description
        
    Returns:
        type: Description
        
    Raises:
        ExceptionType: When this happens
    """
    pass
```

### Testing

- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Test edge cases

```python
def test_feature_does_something():
    """Test that feature does X when Y."""
    # Arrange
    setup_code()
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected
```

## Project Structure

```
src/
├── bacteria/      # Bacteria simulation
├── agent/         # RL agents
├── environment/   # RL environment
└── utils/         # Utilities

tests/
├── unit/          # Unit tests
└── integration/   # Integration tests

examples/          # Example scripts
docs/              # Documentation
config/            # Configuration files
```

## Areas for Contribution

### High Priority

- [ ] Additional RL algorithms (PPO, A3C, etc.)
- [ ] More sophisticated bacteria models
- [ ] Better visualization tools
- [ ] Performance optimizations
- [ ] More comprehensive tests

### Documentation

- [ ] Tutorial notebooks
- [ ] Video tutorials
- [ ] Use case examples
- [ ] API improvements

### Features

- [ ] Multi-objective RL
- [ ] Transfer learning
- [ ] Population visualization
- [ ] Real-time monitoring
- [ ] Hyperparameter tuning tools

### Research

- [ ] Benchmark different RL algorithms
- [ ] Study resistance evolution patterns
- [ ] Compare dosing strategies
- [ ] Analyze reward functions

## Testing Guidelines

### Unit Tests

Test individual components in isolation:

```python
def test_bacteria_cell_resistance():
    """Test resistance is properly bounded."""
    cell = BacteriaCell(resistance=1.5)
    assert cell.resistance <= 1.0
```

### Integration Tests

Test component interactions:

```python
def test_environment_step():
    """Test environment step with agent action."""
    env = AntibioticEnv()
    obs, _ = env.reset()
    obs, reward, done, truncated, info = env.step(0)
    assert isinstance(reward, (int, float))
```

## Documentation Guidelines

- Update README.md for major features
- Add docstrings to all public APIs
- Create examples for new features
- Update API reference as needed
- Include inline comments for complex logic

## Performance Considerations

- Profile code for bottlenecks
- Use vectorized operations (NumPy)
- Avoid unnecessary copies
- Cache expensive computations
- Consider memory usage

## Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS)
- Relevant code snippets
- Error messages/stack traces

## Feature Requests

Include:
- Clear description of the feature
- Use cases and motivation
- Proposed implementation (if any)
- Potential alternatives considered

## Code Review Process

All PRs will be reviewed for:
- Code quality and style
- Test coverage
- Documentation
- Performance impact
- Breaking changes

## Questions?

- Open an issue for discussion
- Join our community discussions
- Check existing issues/PRs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing!
