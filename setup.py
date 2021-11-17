from setuptools import setup


setup(
    name='qiskit_tricks',
    packages=['qiskit_tricks'],
    python_requires='>=3.9',
    install_requires=[
        'qiskit',
        'numpy',
        'pandas',
    ],
)
