from setuptools import setup, find_packages

setup(
    name="fiber_analysis",              # Your package name
    version="0.1.0",                    # Start with 0.1.0 for your initial release
    author="Your Name",                 # Replace with your name
    author_email="you@example.com",     # (Optional) your email
    description="Fiber photometry analysis pipeline",
    packages=find_packages(),           # Automatically find subpackages
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "tdt"
    ],
    entry_points={
        "console_scripts": [
            "fiber-run = main:cli",    # Makes a `fiber-run` command available
        ],
    },
    python_requires=">=3.7",            # Minimum Python version
)
