#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// 读取文件并提取每行第一个以逗号分隔的字符串
std::vector<std::string> getFirstElements(const std::string& filePath) {
    std::vector<std::string> firstElements;
    std::ifstream file(filePath);  // 打开文件

    if (!file.is_open()) {  // 检查文件是否成功打开
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return firstElements;
    }

    std::string line;
    while (std::getline(file, line)) {  // 逐行读取文件
        size_t commaPos = line.find(',');  // 查找第一个逗号的位置
        if (commaPos != std::string::npos) {  // 如果找到逗号
            // 提取从开头到第一个逗号前的子字符串
            std::string firstPart = line.substr(0, commaPos);
            firstElements.push_back(firstPart);
        } else {
            // 如果没有逗号，将整行作为第一个元素（根据实际需求调整）
            firstElements.push_back(line);
        }
    }

    file.close();  // 关闭文件
    return firstElements;
}

int main() {
    std::string filePath = "/data/seery/src/log/data.txt";  // 替换为你的txt文件路径
    std::vector<std::string> results = getFirstElements(filePath);

    // 输出结果
    for (const auto& str : results) {
        std::cout << str << std::endl;
    }

    return 0;
}