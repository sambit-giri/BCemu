def pytest_configure(config):
    config.addinivalue_line(
        "markers", "bcemu2021: tests that require the BCemu2021 emulator files"
    )
    config.addinivalue_line(
        "markers", "bcemu2025: tests that require the BCemu2025 emulator files"
    )
