from setuptools import find_packages, setup

setup(
    name="g1_sonic_tt",
    version="0.1.0",
    description="HITTER-style table-tennis training for the Unitree G1 with SONIC token actions",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy", "torch"],
    extras_require={"deploy": ["pyzmq", "msgpack"]},
)
