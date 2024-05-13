#include <iostream>
#include <vector>
#include "matrix.hpp"

class RiskMeasures {
private:
    Matrix returns;
    Matrix weights;
public:
    RiskMeasures(Matrix returns, Matrix weights) : returns(returns), weights(weights) {
        checkWeights();
    }

    void covariance() {

    }

private:
    void checkWeights() {
        if (weights.getRows() != 1 && weights.getCols() != 3) {
            throw std::invalid_argument("Invalid weights matrix dimensions.");
        }
        double sum = weights.sum();
        if (sum != 1) {
            throw std::invalid_argument("Weights do not sum to 1.");
        }
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

    Matrix y = matrix1(1, 2);
    std::cout << y << std::endl;
    return 0;

}