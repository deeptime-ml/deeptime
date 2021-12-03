from conans import ConanFile


class DeeptimeTests(ConanFile):
    options = {}
    name = "DeeptimeTests"
    version = "0.1"
    requires = (
        "catch2/2.13.7"
    )
    generators = "cmake", "gcc", "txt", "cmake_find_package"
