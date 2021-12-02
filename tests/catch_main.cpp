#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <string>
#include <sstream>
#include <filesystem>
#include <cstdlib>

constexpr char osPathsep() {
#ifdef _WIN32
    return ';';
#else
    return ':';
#endif
}

std::string find_in_path(const std::string &file) {
    std::string path(getenv("PATH"));

    for(std::size_t pos = 0; pos != std::string::npos; pos = path.find(';')) {
        auto nextPos = path.find(osPathsep(), pos+1);
        auto searchPath = path.substr(pos, nextPos == std::string::npos ? path.size() - pos : nextPos-pos);
        for(const auto &f : std::filesystem::directory_iterator(searchPath)) {
            if(file == f.path().filename().string()) {
                return f.path().string();
            }
        }
    }

    return "";
}

void setupPythonHome() {
    #ifdef _WIN32
    std::string pythonExec {"python.exe"};
    #else
    std::string pythonExec {"python"};
    #endif
    auto path = find_in_path(pythonExec);
    if (!path.empty()) {
        std::cerr << "python found in: " << path << "\n";
        setenv("PYTHONHOME", path.c_str(), false);
    } else {
        std::cerr << "python not found in path" << "\n";
    }
}

int main(int argc, char **argv) {
    Catch::Session session;
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) return returnCode;
    return session.run();
}
