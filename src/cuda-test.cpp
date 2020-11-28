#include <iostream>
#include <map>
#include <string>

using std::string;
int matrixMult(int argc, char** argv);
int matrixAdd(int argc, char** argv);
int shuffle(int argc, char** argv);
int sorting(int argc, char** argv);
int misc(int argc, char** argv);
int add(int argc, char** argv);

int main(int argc, char** argv)
{
    std::map<std::string, int(*)(int, char**)> Tests{
        {"m*",      matrixMult},
        {"m+",      matrixAdd},
        {"shuffle", shuffle},
        {"misc",    misc},
        {"add",     add}
    };
    if (1 == argc) {
        std::string keywords;
        for (auto kv : Tests) {
            if (!keywords.empty()) keywords += " ";
            keywords += kv.first;
        }
        std::cout << "valid keywords are: " << keywords << std::endl;
        return 0;
    }
    return (*Tests[argv[1]])(argc - 1, argv + 2);
}