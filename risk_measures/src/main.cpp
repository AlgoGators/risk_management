#include <iostream>
#include <vector>
#include "matrix.hpp"

class RiskMeasures {
private:
    Matrix returns;
    Matrix weights;
public:
    RiskMeasures(Matrix returns, Matrix weights) : returns(returns), weights(weights) {}

    void covariance() {

    }
};

int main() {
    Matrix matrix1 = Matrix({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    Matrix matrix2 = Matrix(std::vector<std::vector<double>>{{7.0, 8.0, 9.0}});
    Matrix matrix3 = Matrix({std::string{"[[7],[8],[9]]"}});
    // std::string matrixStr = "[[0,1,2],[3,4,5],[6,7,8]]";
    // Matrix matrix2 = Matrix(matrixStr);
    Matrix x = matrix2 % matrix1 % matrix3;
    x = x / 2;
    matrix1 = ~matrix1;
    std::string str = matrix1.toString();
    std::cout << str << std::endl;
}