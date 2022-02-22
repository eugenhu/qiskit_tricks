from setuptools import setup


setup(
    name='qiskit_tricks',
    packages=[
        'qiskit_tricks',
    ],
    python_requires='>=3.7',
    install_requires=[
        'qiskit',
        'numpy',
        'pandas',
    ],
)
