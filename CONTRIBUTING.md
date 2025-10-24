# Contributing Guide

Thank you for your interest in the X-Pert project! We welcome various forms of contributions, including but not limited to:

- Reporting bugs
- Suggesting new features
- Submitting code fixes
- Improving documentation
- Sharing use cases

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/X-Pert.git
cd X-Pert
```

### 2. Create Development Environment

```bash
# Using conda
conda env create -f environment.yml
conda activate xpert

# Or using pip
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
pip install -e ".[dev]"
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

## Development Workflow

### 1. Create Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Development

- Write code
- Add tests
- Update documentation
- Ensure code passes all checks

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/unit/test_validation.py

# Run tests with coverage report
pytest --cov=xpert --cov-report=html
```

### 4. Code Quality Checks

```bash
# Code formatting
black xpert tests/

# Import sorting
isort xpert tests/

# Code linting
flake8 xpert tests/

# Type checking
mypy xpert
```

### 5. Commit Changes

```bash
git add .
git commit -m "Describe your changes"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Standards

### Python Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Code linting
- **mypy**: Type checking

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Short description of the example function.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter, defaults to 10
        
    Returns:
        Description of the return value
        
    Raises:
        ValueError: When parameters are invalid
        
    Example:
        >>> example_function("hello", 5)
        True
    """
    pass
```

### Testing Standards

- All new features must have corresponding tests
- Test coverage should be maintained above 80%
- Use descriptive test names
- Tests should be independent and repeatable

### Commit Message Standards

Use clear commit messages:

```
Type(scope): Short description

Detailed description (optional)

Related issue: #123
```

Types include:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation update
- `style`: Code formatting changes
- `refactor`: Code refactoring
- `test`: Test related
- `chore`: Build process or auxiliary tool changes

## Reporting Issues

### Bug Reports

When reporting bugs, please include the following information:

1. Operating system and Python version
2. X-Pert version
3. Steps to reproduce
4. Expected behavior
5. Actual behavior
6. Error messages (if any)

### Feature Requests

When proposing feature requests, please include:

1. Feature description
2. Use cases
3. Expected behavior
4. Possible implementation approach (optional)

## Documentation Contributions

### Updating Documentation

- Synchronize documentation updates with code changes
- Use clear and concise language
- Provide complete example code
- Ensure all links are valid

### Building Documentation

```bash
cd docs
make html
```

## Release Process

### Version Number Standards

We use Semantic Versioning:

- `MAJOR`: Incompatible API changes
- `MINOR`: Backward-compatible functionality additions
- `PATCH`: Backward-compatible bug fixes

### Release Steps

1. Update version number
2. Update CHANGELOG.md
3. Create release tag
4. Automatically publish to PyPI

## Community Guidelines

### Code of Conduct

- Be friendly and respectful
- Welcome different perspectives and experience levels
- Focus on what's best for the community
- Respect different viewpoints, ideas, and skill levels
- Accept constructive criticism
- Focus on community members and the community as a whole

### Getting Help

- Check documentation and FAQ
- Ask questions in GitHub Discussions
- Send email to: your.email@example.com

## License

By contributing code, you agree that your contributions will be released under the MIT License.

## Acknowledgments

Thanks to all contributors for their efforts! Your contributions make X-Pert better.